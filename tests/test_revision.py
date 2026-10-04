from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.ravdess import DATASET_FACTORY, RavdessAVDatasetWavLM, split_pairs_by_actor, PairRecord
from models.chumachenko_ia import ChumachenkoIntermediateAttentionFusion
from revision.audit import audit_dataset, expected_keys, require_complete
from revision.aggregate import aggregate, METRICS
from revision.protocol import fixed_folds, validate_split, validate_warm_start, run_path, PROTOCOL
from revision.run import run, build_arg_parser as revision_parser
from revision.profiles import production_profile, resolve_profiles
from revision.protocol import expected_run_config, validate_completed_run
from train import EmotionTrainer, build_arg_parser
from eval import EmotionEvaluator, build_arg_parser as eval_parser
from utils.metrics import classification_metrics


def fixture_revision(root, fold=1, smoke=True):
    settings = {"seed":42,"frames":2 if smoke else 8,"use_face_crop":not smoke,
                "num_workers":0 if smoke else -1}
    return {"output_root":str(root),"data_root":str(root / "data"),"protocol":PROTOCOL,
            "split":fixed_folds()[fold-1],"smoke":smoke,"settings":settings,
            "profiles":resolve_profiles(smoke,settings),"source_fingerprint":"test-source",
            "dataset_fingerprint":"test-data"}


def write_completed_run(path, config, score=0.2):
    path.mkdir(parents=True,exist_ok=True)
    (path.parent/"split.json").write_text(json.dumps(config["revision"]))
    (path/"config.json").write_text(json.dumps(config))
    best = {"model":{"weight":torch.ones(1)},"config":config,"epoch":1,"val_f1":0.3}
    torch.save(best,path/"best.pt")
    record = {"config":config,"test":{k:score for k in METRICS},"best_epoch":1,
              "val_macro_f1":0.3,"checkpoint":str((path/"best.pt").resolve())}
    (path/"metrics.json").write_text(json.dumps(record))
    return record


class AudioFixture(torch.nn.Module):
    sequence_dim = 4
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 4)
    def encode_sequence(self, audio):
        return self.proj(audio.transpose(1, 2))


class VideoFixture(torch.nn.Module):
    embedding_dim = 4
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.AdaptiveAvgPool2d(1))


class RevisionTests(unittest.TestCase):
    def test_fixed_schedule_coverage_and_disjointness(self):
        folds = fixed_folds()
        self.assertEqual(len(folds), 6)
        for i, split in enumerate(folds):
            self.assertEqual(split["test_actors"], list(range(4*i+1, 4*i+5)))
            self.assertEqual(split["val_actors"], list(range(4*((i+1)%6)+1, 4*((i+1)%6)+5)))
            validate_split(split)
        for partition in ("val", "test"):
            self.assertEqual(sorted(a for f in folds for a in f[f"{partition}_actors"]), list(range(1,25)))
        for corruption in ("overlap", "duplicate", "missing", "count"):
            split = copy.deepcopy(folds[0])
            if corruption == "overlap": split["train_actors"][0] = 1
            if corruption == "duplicate": split["val_actors"][0] = 6
            if corruption == "missing": split["test_actors"][0] = 25
            if corruption == "count": split["test_actors"].pop()
            with self.assertRaises(ValueError): validate_split(split)

    def test_complete_and_corrupted_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for key in expected_keys():
                suffix = "-".join(f"{n:02d}" for n in key)
                for modality, extension in ((2,"mp4"), (3,"wav")):
                    (root / f"{modality:02d}-{suffix}.{extension}").touch()
            report = audit_dataset(root)
            require_complete(report)
            self.assertEqual(report["total_pairs"], 1440)
            self.assertEqual(report["actors"], list(range(1,25)))
            self.assertEqual(report["emotions"], {str(e): 96 if e==1 else 192 for e in range(1,9)})
            for actor in report["by_actor"].values():
                self.assertEqual([actor[x] for x in ("audio","video","pairs")], [60,60,60])
                self.assertEqual(actor["emotions"], {str(e):4 if e==1 else 8 for e in range(1,9)})
            victim = root / "03-01-01-01-01-01-01.wav"
            victim.rename(root / "03-01-01-02-01-01-01.wav")
            bad = audit_dataset(root)
            self.assertFalse(bad["complete"])
            self.assertTrue(bad["invalid_files"])
            self.assertTrue(bad["missing_counterparts"]["audio"])
            with self.assertRaises(ValueError): require_complete(bad)
            victim = root / "03-01-01-02-01-01-01.wav"
            victim.rename(root / "03-01-01-01-01-01-01.wav")
            duplicate = root / "duplicate"
            duplicate.mkdir()
            (duplicate / "03-01-01-01-01-01-01.wav").touch()
            bad = audit_dataset(root)
            self.assertEqual(len(bad["duplicates"]), 1)
            with self.assertRaises(ValueError): require_complete(bad)

    def test_paths_and_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture_revision(root)
            audio = run_path(root,1,"audio") / "best.pt"
            self.assertNotEqual(audio,run_path(root,2,"audio") / "best.pt")
            config = expected_run_config(revision,"audio")
            write_completed_run(audio.parent,config)
            validate_warm_start(audio,revision,"audio")
            for bad_path in (root/"best_audio.pt",run_path(root,2,"audio")/"best.pt"):
                with self.assertRaises(ValueError): validate_warm_start(bad_path,revision,"audio")
            for changed in (None,{**revision,"split":fixed_folds()[1]},{**revision,"smoke":False}):
                checkpoint = torch.load(audio,weights_only=True)
                checkpoint["config"]["revision"] = changed
                torch.save(checkpoint,audio)
                with self.assertRaises(ValueError): validate_warm_start(audio,revision,"audio")
                write_completed_run(audio.parent,config)

    def test_ia_shape_backward_and_query_sum_semantics(self):
        torch.manual_seed(1)
        model = ChumachenkoIntermediateAttentionFusion(AudioFixture(),VideoFixture(),d_model=4)
        self.assertEqual(model.num_heads,1)
        logits = model(torch.randn(2,3,3,8,8),torch.randn(2,1,7))
        self.assertEqual(tuple(logits.shape),(2,8))
        logits.square().mean().backward()
        for layer in (model.audio_query,model.video_key,model.video_query,model.audio_key):
            self.assertIsNotNone(layer.weight.grad)
            self.assertGreater(layer.weight.grad.abs().sum().item(),0)
        # Equal attention logits => query-count/key-count modulation (not constant sum over keys).
        with torch.no_grad():
            for layer in (model.audio_query,model.video_key,model.video_query,model.audio_key):
                layer.weight.zero_()
        a, v = model.modulate(torch.ones(1,6,4),torch.ones(1,3,4))
        torch.testing.assert_close(a,torch.full_like(a,0.5))
        torch.testing.assert_close(v,torch.full_like(v,2.0))

    def test_best_checkpoint_reloaded_before_single_test_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_arg_parser().parse_args(["--data_root",tmp,"--epochs","2","--fusion","video"])
            args.output_dir = tmp
            model = torch.nn.Linear(1,8)
            loaders = [object(),object(),object()]
            count = 0
            def training(*unused, **kwargs):
                nonlocal count
                count += 1
                with torch.no_grad(): model.weight.fill_(count)
                return {"loss":0.,"cls_loss":0.,"contrastive_loss":0.,"acc":0.,"f1":0.}
            scores = []
            def evaluation(m,loader,*unused,**kwargs):
                if loader is loaders[1]:
                    f1 = 0.9 if count==1 else 0.1
                else:
                    scores.append(m.weight.detach().clone())
                    f1 = 0.2
                return {"loss":0.,"cls_loss":0.,"contrastive_loss":0.,"acc":0.,"f1":f1,
                        "accuracy":0.,"precision":0.,"recall":0.,"macro_f1":f1}
            with mock.patch("train.build_dataloaders",return_value=(*loaders,{"test":1})), \
                 mock.patch("train.build_model",return_value=model), \
                 mock.patch("train.train_one_epoch",side_effect=training), \
                 mock.patch("train.evaluate",side_effect=evaluation), \
                 mock.patch("train.torch.cuda.is_available",return_value=False), \
                 mock.patch("sys.stdout",new_callable=io.StringIO):
                EmotionTrainer(args).run()
            self.assertEqual(len(scores),1)
            torch.testing.assert_close(scores[0],torch.ones_like(scores[0]))
            record = json.loads((Path(tmp)/"metrics.json").read_text())
            self.assertEqual(record["best_epoch"],1)
            self.assertEqual(Path(record["checkpoint"]).name,"best_video.pt")

    def test_evaluator_checkpoint_controls_waveform_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"best.pt"
            model = torch.nn.Linear(1,8)
            torch.save({"model":model.state_dict(),"config":{"use_wavlm":True,"fusion":"audio",
                "frames":2,"num_classes":8,"use_face_crop":False}},path)
            args = eval_parser().parse_args(["--data_root",tmp,"--checkpoint",str(path),"--num_workers","0"])
            pair = PairRecord(Path('v'),Path('a'),1,1,1,1,22)
            with mock.patch("eval.PAIR_SERVICE.build_pairs",return_value=[pair]), \
                 mock.patch("eval.DATASET_FACTORY.create",wraps=DATASET_FACTORY.create) as factory, \
                 mock.patch("eval.build_model",return_value=model), \
                 mock.patch("eval.evaluate",return_value={}), \
                 mock.patch("eval.torch.cuda.is_available",return_value=False):
                EmotionEvaluator(args).run()
            self.assertTrue(factory.call_args.kwargs["use_wavlm"])
            self.assertFalse(factory.call_args.kwargs["use_face_crop"])
            self.assertEqual(factory.call_args.kwargs["num_frames"],2)
            self.assertIsInstance(DATASET_FACTORY.create([pair],use_wavlm=True),RavdessAVDatasetWavLM)
            conflict = eval_parser().parse_args(["--data_root",tmp,"--checkpoint",str(path),"--fusion","video"])
            with self.assertRaises(ValueError): EmotionEvaluator(conflict).run()

    def test_metrics_fixed_eight_class_macro(self):
        scores = classification_metrics(torch.tensor([0,0]),torch.tensor([0,1]))
        self.assertEqual(scores["accuracy"],0.5)
        self.assertAlmostEqual(scores["precision"],0.5/8)
        self.assertAlmostEqual(scores["recall"],1/8)
        self.assertAlmostEqual(scores["macro_f1"],(2/3)/8)

    def test_aggregate_mean_sample_std_partial_and_mixing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for split in fixed_folds():
                path = run_path(root,split["fold"],"audio") / "metrics.json"
                path.parent.mkdir(parents=True)
                config = expected_run_config(fixture_revision(root,split["fold"],False),"audio")
                write_completed_run(path.parent,config,score=split["fold"]/10)
                paths.append(path)
            result = aggregate(root)["methods"]["audio"]
            self.assertTrue(result["complete_six_folds"])
            self.assertAlmostEqual(result["summary"]["accuracy"]["mean"],0.35)
            self.assertAlmostEqual(result["summary"]["accuracy"]["std"],math.sqrt(0.035))
            corrupted = json.loads(paths[-1].read_text())
            corrupted["config"]["revision"]["smoke"] = True
            paths[-1].write_text(json.dumps(corrupted))
            with self.assertRaises(ValueError): aggregate(root)
            for path in paths[1:]: path.unlink()
            result = aggregate(root)["methods"]["audio"]
            self.assertFalse(result["complete_six_folds"])
            self.assertIsNone(result["summary"]["accuracy"]["std"])

    def test_resume_completed_fold_then_remaining_folds_and_skip_all(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(frames=2, output_root=Path(tmp)/"out",
                smoke=True, data_root=None, fold=1, model="all", seed=42,
                no_face_crop=True, num_workers=0)
            configs = []
            def fake_train(train_args):
                configs.append(train_args)
                path = Path(train_args.output_dir)
                record = write_completed_run(path,vars(train_args))
                return mock.Mock(run=lambda: record["test"])
            with mock.patch("revision.run.EmotionTrainer", side_effect=fake_train), \
                 mock.patch("sys.stdout",new_callable=io.StringIO):
                first = run(args)
                self.assertEqual(len(configs),6)
                self.assertFalse(first["methods"]["audio"]["complete_six_folds"])
                original_metrics = (run_path(args.output_root,1,"audio")/"metrics.json").read_bytes()
                args.fold = None
                result = run(args)
                self.assertEqual(original_metrics,(run_path(args.output_root,1,"audio")/"metrics.json").read_bytes())
                self.assertEqual(len(configs),36)
                self.assertEqual(len(result["methods"]),6)
                self.assertTrue(all(m["complete_six_folds"] for m in result["methods"].values()))
                for config in configs:
                    fold = config.revision["split"]["fold"]
                    self.assertEqual(config.split_mode,"actor")
                    self.assertTrue(config.use_wavlm)
                    if config.fusion not in {"audio","video"}:
                        for modality in ("audio","video"):
                            self.assertEqual(Path(getattr(config,f"{modality}_ckpt")),
                                run_path(args.output_root,fold,modality)/"best.pt")
                run(args)
                self.assertEqual(len(configs),36)  # Every completed run is skipped.
                args.seed = 43
                with self.assertRaises(ValueError): run(args)
                args.smoke = False
                with self.assertRaises(ValueError): run(args)

    def test_canonical_production_profiles_and_ids(self):
        audio = production_profile("audio")
        self.assertEqual(audio["id"],"spmb2026-audio-v1")
        a = audio["config"]
        self.assertEqual((a["epochs"],a["batch_size"],a["lr"],a["weight_decay"]),(20,16,1e-3,1e-4))
        self.assertEqual((a["wavlm_stage"],a["backbone_lr"]),(2,3e-5))
        for name in ("audio","video","gated","concat","chumachenko_ia","xattn"):
            c = production_profile(name)["config"]
            self.assertEqual(c["num_workers"],-1)
            self.assertTrue(c["use_cosine_annealing"])
            self.assertTrue(c["use_wavlm"])
            self.assertFalse(c["smoke"])
        video = production_profile("video")["config"]
        self.assertEqual((video["epochs"],video["batch_size"],video["lr"],video["weight_decay"],
                          video["early_stopping_patience"]),(20,16,1e-3,1e-4,10))
        for name in ("gated","concat","chumachenko_ia"):
            c = production_profile(name)["config"]
            self.assertEqual((c["epochs"],c["batch_size"],c["lr"],c["weight_decay"]),(30,8,3e-4,1e-4))
            self.assertTrue(c["two_stage_training"])
            self.assertEqual(c["stage1_epochs"],5)
            self.assertEqual((c["audio_backbone_lr"],c["video_backbone_lr"]),(1e-5,1e-5))
            self.assertEqual((c["fusion_unfreeze_wavlm_layers"],c["fusion_unfreeze_video_blocks"]),(2,1))
            self.assertEqual(c["early_stopping_patience"],8)
        x = production_profile("xattn")["config"]
        self.assertEqual((x["epochs"],x["batch_size"],x["lr"],x["weight_decay"]),(35,8,2e-4,2e-4))
        self.assertEqual((x["xattn_head"],x["xattn_d_model"],x["xattn_heads"]),("gated",96,4))
        self.assertEqual((x["label_smoothing"],x["stage1_epochs"]),(0.05,6))
        self.assertTrue(x["two_stage_training"])
        self.assertEqual((x["audio_backbone_lr"],x["video_backbone_lr"]),(8e-6,8e-6))
        self.assertEqual((x["xattn_attn_dropout"],x["xattn_stochastic_depth"]),(0.1,0.1))
        self.assertEqual((x["fusion_unfreeze_wavlm_layers"],x["fusion_unfreeze_video_blocks"]),(2,1))
        self.assertEqual(x["early_stopping_patience"],10)
        self.assertEqual(revision_parser().parse_args([]).num_workers,-1)
        # Copies must not allow IA/concat changes to mutate the canonical gated profile.
        production_profile("chumachenko_ia")["config"]["lr"] = 99
        self.assertEqual(production_profile("gated")["config"]["lr"],3e-4)

    def test_production_runner_applies_full_profiles_across_all_folds_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = revision_parser().parse_args(["--data-root",str(Path(tmp)/"data"),
                                                 "--output-root",str(Path(tmp)/"out")])
            configs = []
            def fake_train(train_args):
                configs.append(train_args)
                record = write_completed_run(Path(train_args.output_dir),vars(train_args))
                return mock.Mock(run=lambda:record["test"])
            # Controlled audit and trainer mocks: no real media or model training is performed.
            with mock.patch("revision.run.audit_dataset",return_value={"complete":True,"fingerprint":"mock-data"}), \
                 mock.patch("revision.run.EmotionTrainer",side_effect=fake_train), \
                 mock.patch("sys.stdout",new_callable=io.StringIO):
                result = run(args)
                self.assertEqual(len(configs),36)
                for config in configs:
                    expected = production_profile(config.fusion)
                    self.assertEqual(config.profile_id,expected["id"])
                    self.assertEqual(config.resolved_profile,expected["config"])
                    for key,value in expected["config"].items():
                        self.assertEqual(getattr(config,key),value)
                self.assertTrue(all(m["complete_six_folds"] for m in result["methods"].values()))
                run(args)
                self.assertEqual(len(configs),36)

    def test_resume_rejects_incomplete_or_mismatched_artifacts_before_training(self):
        corruptions = ("best.pt","metrics.json","config.json","split.json","metric-config",
                       "checkpoint-config","profile-id","resolved-profile","source","dataset",
                       "checkpoint-path","epoch","test-score","unreadable-checkpoint","empty-state",
                       "coordinated-config","other-incomplete-fold")
        for corruption in corruptions:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                args = revision_parser().parse_args(["--smoke","--fold","1","--model","audio",
                                                     "--output-root",str(Path(tmp)/"out")])
                def fake_train(train_args):
                    record = write_completed_run(Path(train_args.output_dir),vars(train_args))
                    return mock.Mock(run=lambda:record["test"])
                with mock.patch("revision.run.EmotionTrainer",side_effect=fake_train), \
                     mock.patch("sys.stdout",new_callable=io.StringIO):
                    run(args)
                path = run_path(args.output_root,1,"audio")
                if corruption == "other-incomplete-fold":
                    folder = args.output_root/"fold_02"
                    folder.mkdir()
                    (folder/"audio").mkdir()
                elif corruption == "empty-state":
                    obj = torch.load(path/"best.pt",weights_only=True)
                    obj["model"] = {}
                    torch.save(obj,path/"best.pt")
                elif corruption == "coordinated-config":
                    obj = json.loads((path/"config.json").read_text())
                    obj["lr"] = 99
                    (path/"config.json").write_text(json.dumps(obj))
                    record = json.loads((path/"metrics.json").read_text())
                    record["config"] = obj
                    (path/"metrics.json").write_text(json.dumps(record))
                    ckpt = torch.load(path/"best.pt",weights_only=True)
                    ckpt["config"] = obj
                    torch.save(ckpt,path/"best.pt")
                elif corruption in ("best.pt","metrics.json","config.json"):
                    (path/corruption).unlink()
                elif corruption == "split.json":
                    (path.parent/corruption).unlink()
                elif corruption == "unreadable-checkpoint":
                    (path/"best.pt").write_text("invalid checkpoint")
                elif corruption == "checkpoint-config":
                    obj = torch.load(path/"best.pt",weights_only=True)
                    obj["config"]["wavlm_stage"] = 1
                    torch.save(obj,path/"best.pt")
                else:
                    obj = json.loads((path/"metrics.json").read_text())
                    if corruption == "metric-config": obj["config"]["lr"] = 99
                    if corruption == "profile-id": obj["config"]["profile_id"] = "wrong"
                    if corruption == "resolved-profile": obj["config"]["resolved_profile"]["wavlm_stage"] = 1
                    if corruption == "source": obj["config"]["revision"]["source_fingerprint"] = "wrong"
                    if corruption == "dataset": obj["config"]["revision"]["dataset_fingerprint"] = "wrong"
                    if corruption == "checkpoint-path": obj["checkpoint"] = "old/best_audio.pt"
                    if corruption == "epoch": obj["best_epoch"] = 9
                    if corruption == "test-score": obj["test"]["accuracy"] = float("nan")
                    (path/"metrics.json").write_text(json.dumps(obj))
                with mock.patch("revision.run.EmotionTrainer") as trainer:
                    with self.assertRaises(ValueError): run(args)
                    trainer.assert_not_called()

    def test_pretrained_wavlm_failure_does_not_silently_randomize(self):
        from models.wavlm_audio import WavLMAudioEncoder
        with mock.patch("models.wavlm_audio.WavLMModel.from_pretrained",side_effect=OSError("offline")):
            with self.assertRaises(OSError): WavLMAudioEncoder(8)

    def test_checkpoint_initialization_needs_no_pretrained_download(self):
        from types import SimpleNamespace
        from models.wavlm_audio import WavLMAudioEncoder
        backbone = torch.nn.Linear(1,1)
        backbone.config = SimpleNamespace(hidden_size=768)
        with mock.patch("models.wavlm_audio.WavLMModel.from_pretrained") as download, \
             mock.patch("models.wavlm_audio.WavLMModel",return_value=backbone) as constructor:
            # Patch constructor's from_pretrained after replacement for a meaningful assertion.
            encoder = WavLMAudioEncoder(8,checkpoint_init=True)
            constructor.from_pretrained.assert_not_called()
            self.assertIs(encoder.wavlm,backbone)

    def test_production_runner_refuses_incomplete_data_before_training(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(epochs=1,batch_size=4,frames=2,output_root=Path(tmp)/"out",
                                   smoke=False,data_root=Path(tmp)/"missing")
            with mock.patch("revision.run.EmotionTrainer") as trainer:
                with self.assertRaises(ValueError): run(args)
                trainer.assert_not_called()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
