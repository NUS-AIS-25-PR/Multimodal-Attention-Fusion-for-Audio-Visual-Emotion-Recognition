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
from revision.run import run
from train import EmotionTrainer, build_arg_parser
from eval import EmotionEvaluator, build_arg_parser as eval_parser
from utils.metrics import classification_metrics


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
            split = fixed_folds()[0]
            revision = {"output_root": str(root), "split":split, "protocol": PROTOCOL, "smoke":False}
            audio = run_path(root, 1, "audio") / "best.pt"
            self.assertNotEqual(audio, run_path(root, 2, "audio") / "best.pt")
            audio.parent.mkdir(parents=True)
            checkpoint = {"config": {"revision":revision, "fusion":"audio", "use_wavlm":True}}
            torch.save(checkpoint, audio)
            validate_warm_start(audio, revision, "audio")
            for bad_path in (root / "best_audio.pt", run_path(root,2,"audio") / "best.pt"):
                with self.assertRaises(ValueError): validate_warm_start(bad_path, revision, "audio")
            for changed in (None, {**revision,"split":fixed_folds()[1]}, {**revision,"smoke":True}):
                checkpoint["config"]["revision"] = changed
                torch.save(checkpoint, audio)
                with self.assertRaises(ValueError): validate_warm_start(audio, revision, "audio")

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
                record = {"config":{"fusion":"audio","revision":{"protocol":PROTOCOL,"split":split,"smoke":False}},
                          "test":{k:split["fold"]/10 for k in METRICS}}
                path.write_text(json.dumps(record)); paths.append(path)
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

    def test_runner_all_folds_models_paths_and_overwrite_guard(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(epochs=1, batch_size=4, frames=2, output_root=Path(tmp)/"out",
                smoke=True, data_root=None, fold=None, model="all", seed=42, lr=0.001,
                no_face_crop=True, num_workers=0)
            configs = []
            def fake_train(train_args):
                configs.append(train_args)
                path = Path(train_args.output_dir)
                record = {"config":vars(train_args), "test":{k:0.2 for k in METRICS}}
                (path/"metrics.json").write_text(json.dumps(record))
                torch.save({"config":vars(train_args),"model":{}},path/"best.pt")
                return mock.Mock(run=lambda: record["test"])
            with mock.patch("revision.run.EmotionTrainer", side_effect=fake_train), \
                 mock.patch("sys.stdout",new_callable=io.StringIO):
                result = run(args)
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
                with self.assertRaises(FileExistsError): run(args)
                args.epochs = 2  # Smoke always overrides epochs to 1.
                args.lr = 0.002
                with self.assertRaises(ValueError): run(args)
                args.smoke = False
                with self.assertRaises(ValueError): run(args)

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
