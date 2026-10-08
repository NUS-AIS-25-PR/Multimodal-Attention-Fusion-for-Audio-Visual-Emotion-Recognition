from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

from test_revision import fixture_revision, write_completed_run
from revision.artifacts import ARTIFACT_SCHEMA, HistoryWriter, read_history, read_predictions
from revision.figures import (comparison_summary, generate_figures, load_records,
                              normalize_counts, pooled_counts)
from revision.protocol import expected_run_config, validate_completed_run
from revision.run import build_arg_parser, run
from revision.tracking import RunTracker, TrackingOptions
from train import evaluate


def completed(root, fold=1, model="audio"):
    revision = fixture_revision(root, fold)
    revision["artifact_schema"] = ARTIFACT_SCHEMA
    config = expected_run_config(revision, model)
    path = root / f"fold_{fold:02d}" / model
    write_completed_run(path, config)
    (root / "mode.json").write_text(json.dumps({"smoke":True,"protocol":revision["protocol"]}))
    return path, config


class TrackingFigureTests(unittest.TestCase):
    def test_disabled_sdk_never_imported_and_history_is_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with mock.patch("revision.tracking.importlib.import_module") as sdk:
                tracker = RunTracker(path, {"fusion":"audio"}, TrackingOptions())
                tracker.log({"epoch":1})
                tracker.finish()
                sdk.assert_not_called()
            scores = {"loss":0.5,"accuracy":0.6,"precision":0.7,"recall":0.8,"macro_f1":0.9}
            history = HistoryWriter(path / "history.csv")
            history.append(1,1,scores,scores,[0.001])
            history.append(2,2,scores,scores,[1e-5,1e-5,3e-4])
            rows = read_history(path / "history.csv")
            self.assertEqual([r["stage"] for r in rows],[1,2])
            self.assertEqual(rows[1]["learning_rates"],[1e-5,1e-5,3e-4])
            self.assertEqual(rows[0]["train_precision"],0.7)
            original = (path / "history.csv").read_bytes()
            with self.assertRaises(FileExistsError): HistoryWriter(path / "history.csv")
            self.assertEqual(original,(path / "history.csv").read_bytes())

    def test_optional_sdk_failure_is_redacted_and_preserves_local_logging(self):
        for failure in (ModuleNotFoundError("private-message"),RuntimeError("private-message")):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                with mock.patch("revision.tracking.importlib.import_module",side_effect=failure), \
                     mock.patch("sys.stdout",new_callable=io.StringIO) as out:
                    tracker=RunTracker(path,{"fusion":"audio"},TrackingOptions(mode="offline"))
                    tracker.log({"epoch":1})
                    tracker.finish()
                self.assertNotIn("private-message",out.getvalue())
                self.assertNotIn("private-message",(path/"tracking.json").read_text())
                self.assertEqual(json.loads((path/"tracking.json").read_text())["status"],"unavailable")
                scores=dict(loss=1.,accuracy=.2,precision=.3,recall=.4,macro_f1=.5)
                HistoryWriter(path/"history.csv").append(1,0,scores,scores,[.001])
                self.assertEqual(len(read_history(path/"history.csv")),1)

    def test_online_offline_metadata_rng_and_scalar_only_transport(self):
        for mode in ("online","offline"):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                revision=fixture_revision(root)
                revision["git_commit"]="12345678abc"
                config=expected_run_config(revision,"gated")
                def consume_rng(*args,**kwargs):
                    random.random(); np.random.rand(); torch.rand(1)
                    return sdk_run
                sdk_run=mock.Mock()
                sdk_run.log.side_effect=consume_rng
                sdk_run.finish.side_effect=consume_rng
                sdk=mock.Mock()
                sdk.init.side_effect=consume_rng
                before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
                with mock.patch("revision.tracking.importlib.import_module",return_value=sdk):
                    tracker=RunTracker(root,config,TrackingOptions(mode=mode))
                    tracker.log({"epoch":1,"train/macro_f1":.5})
                    tracker.finish()
                self.assertEqual(random.getstate(),before[0])
                np.testing.assert_equal(np.random.get_state(),before[1])
                torch.testing.assert_close(torch.get_rng_state(),before[2])
                call=sdk.init.call_args.kwargs
                self.assertEqual(call["mode"],mode)
                self.assertEqual(call["name"],"fold-01-gated")
                self.assertIn("12345678",call["group"])
                self.assertEqual(call["config"]["configuration"],config)
                self.assertEqual(call["config"]["split"],revision["split"])
                self.assertFalse(call["save_code"])
                sdk_run.save.assert_not_called()
                sdk_run.log_artifact.assert_not_called()
                sdk_run.finish.assert_called_once()

    def test_predictions_collected_in_the_single_evaluation_pass(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.calls=0
            def forward(self, audio):
                self.calls+=1
                logits=torch.zeros(len(audio),8); logits[:,0]=1
                return logits
        samples=[(torch.zeros(2,3,8,8),torch.ones(1),i,
                  dict(actor=i+1,emotion=i+1,intensity=1,statement=1,repetition=1)) for i in range(2)]
        model=Model(); predictions=[]
        scores=evaluate(model,torch.utils.data.DataLoader(samples,batch_size=2),torch.device("cpu"),
                        torch.nn.CrossEntropyLoss(),"audio",prediction_rows=predictions)
        self.assertEqual(model.calls,1)
        self.assertEqual(scores["accuracy"],.5)
        self.assertEqual([r["label"] for r in predictions],[0,1])
        self.assertEqual([r["prediction"] for r in predictions],[0,0])
        self.assertEqual([r["actor"] for r in predictions],[1,2])
        self.assertEqual(predictions[1]["sample_id"],"01-02-01-01-01-02")

    def test_new_artifacts_and_held_out_integrity_required_for_completion(self):
        for corruption in ("history.csv","test_predictions.csv","confusion_matrix.json","tracking.json","actor","duplicate",
                           "confusion","score","history-best","history-nan"):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as tmp:
                path,config=completed(Path(tmp))
                validate_completed_run(path,config)
                if corruption.endswith((".csv",".json")):
                    (path/corruption).unlink()
                elif corruption in ("actor","duplicate"):
                    lines=(path/"test_predictions.csv").read_text().splitlines()
                    if corruption=="actor":
                        lines[1]=lines[1].replace(",1,",",5,",1)
                    else:
                        lines.append(lines[1])
                    (path/"test_predictions.csv").write_text("\n".join(lines)+"\n")
                elif corruption=="confusion":
                    obj=json.loads((path/"confusion_matrix.json").read_text());obj["counts"][0][0]+=1
                    (path/"confusion_matrix.json").write_text(json.dumps(obj))
                elif corruption=="score":
                    obj=json.loads((path/"metrics.json").read_text());obj["test"]["accuracy"]=.1
                    (path/"metrics.json").write_text(json.dumps(obj))
                else:
                    text=(path/"history.csv").read_text().replace("0.3", "nan" if corruption=="history-nan" else "0.9")
                    (path/"history.csv").write_text(text)
                with self.assertRaises(ValueError): validate_completed_run(path,config)

    def test_runner_tracking_options_do_not_change_profiles_or_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=build_arg_parser().parse_args(["--smoke","--fold","1","--model","audio",
                 "--output-root",str(Path(tmp)/"out"),"--wandb-mode","offline"])
            configs=[]
            def fake_train(train_args,tracking):
                configs.append(copy.deepcopy(vars(train_args)))
                self.assertEqual(tracking.mode,"offline")
                self.assertFalse(train_args.wandb)
                record=write_completed_run(Path(train_args.output_dir),vars(train_args))
                return mock.Mock(run=lambda:record["test"])
            with mock.patch("revision.run.EmotionTrainer",side_effect=fake_train):
                run(args)
            path=args.output_root/"fold_01/audio"
            before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in args.output_root.rglob('*') if p.is_file()}
            args.wandb_mode="disabled"
            with mock.patch("revision.run.EmotionTrainer") as trainer, mock.patch("train.evaluate") as evaluator:
                run(args)
                trainer.assert_not_called();evaluator.assert_not_called()
            self.assertEqual(before,{str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in args.output_root.rglob('*') if p.is_file()})
            self.assertEqual(configs[0]["resolved_profile"]["wavlm_stage"],2)

    def test_figure_exports_are_marked_and_do_not_modify_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"runs";path,_=completed(root)
            output=Path(tmp)/"figures"
            original={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaisesRegex(ValueError,"Synthetic smoke"):
                generate_figures(root,output)
            self.assertFalse(output.exists())
            manifest=generate_figures(root,output,allow_smoke=True)
            self.assertTrue(manifest["smoke_not_for_paper"])
            self.assertFalse(manifest["models"]["audio"]["complete"])
            self.assertIsNone(manifest["models"]["audio"]["metrics"]["accuracy"]["std"])
            self.assertEqual(len(manifest["figures"]),12)
            self.assertEqual(original,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            for filename in manifest["figures"]:
                file=Path(filename);self.assertGreater(file.stat().st_size,1000)
                if file.suffix==".pdf":self.assertTrue(file.read_bytes().startswith(b'%PDF'))
                elif file.suffix==".svg":
                    self.assertIn("NOT FOR PAPER",file.read_text())
                    self.assertIn("INCOMPLETE",file.read_text())
                else:
                    with Image.open(file) as im:self.assertAlmostEqual(im.info["dpi"][0],300,delta=.1)

    def test_missing_incomplete_or_mixed_figure_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"runs"
            with self.assertRaisesRegex(ValueError,"mode.json"):load_records(root,["audio"])
            path,_=completed(root)
            (path/"history.csv").unlink()
            with self.assertRaises(ValueError):load_records(root,["audio"],True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);completed(root,1);path,config=completed(root,2)
            config["revision"]["dataset_fingerprint"]="different"
            # A second internally consistent completed run from a different experiment.
            for p in path.iterdir():p.unlink()
            write_completed_run(path,config)
            with self.assertRaisesRegex(ValueError,"mixed"):load_records(root,["audio"],True)

    def test_sample_std_and_pooled_counts_use_only_unique_held_out_rows(self):
        rows=[{"fold":i,"metrics":{"test":{k:i/10 for k in ("accuracy","precision","recall","macro_f1")}}}
              for i in range(1,7)]
        stats=comparison_summary({"audio":rows})["audio"]
        self.assertTrue(stats["complete"])
        self.assertAlmostEqual(stats["metrics"]["accuracy"]["mean"],.35)
        self.assertAlmostEqual(stats["metrics"]["accuracy"]["std"],math.sqrt(.035))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(1,7):completed(root,i)
            loaded=load_records(root,["audio"],True)["records"]["audio"]
            counts=pooled_counts(loaded)
            np.testing.assert_array_equal(counts,np.diag([3]*8))
            np.testing.assert_array_equal(normalize_counts(counts),np.eye(8))
            np.testing.assert_array_equal(normalize_counts(np.zeros((8,8))),np.zeros((8,8)))
            with self.assertRaisesRegex(ValueError,"repeated"):pooled_counts([loaded[0],loaded[0]])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
