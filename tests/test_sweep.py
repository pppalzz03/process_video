from vidanalytics.sweep import expand_configs, recommend, render_markdown, summarize


def test_expand_caps_workers_and_dedupes():
    cfg = expand_configs([60, 10], [1, 4, 8, 16], [False], {60: 3, 10: 18})
    thr = [(c["seg"], c["workers"]) for c in cfg if c["decoder"] == "threads"]
    assert (60, 3) in thr and thr.count((60, 3)) == 1          # 4/8/16 all collapse to 3 workers
    assert (10, 16) in thr and (10, 8) in thr
    assert sum(c["decoder"] == "sequential" for c in cfg) == 1  # one baseline, first seg only


def test_expand_resize_both_doubles_configs():
    cfg = expand_configs([10], [2, 4], [False, True], {10: 18})
    assert len(cfg) == 2 + 4


def test_summarize_median():
    r = summarize([{"processed_fps": 100, "avg_cores_used": 2}, {"processed_fps": 300, "avg_cores_used": 4},
                   {"processed_fps": 200, "avg_cores_used": 3}])
    assert r["processed_fps"] == 200 and r["fps_min"] == 100 and r["fps_max"] == 300 and r["n_runs"] == 3


def row(w, fps, **kw):
    return {"seg": 10, "decoder": "threads", "workers": w, "resize": False, "chunks": 18, "processed_fps": fps,
            "fps_min": fps, "fps_max": fps, "avg_cores_used": w, **kw}


def test_recommend_knee_and_gpu_bound_note():
    rows = [{"seg": 10, "decoder": "sequential", "workers": 0, "resize": False, "chunks": 18, "processed_fps": 199,
             "fps_min": 199, "fps_max": 199, "avg_cores_used": 1},
            row(2, 300), row(4, 480), row(8, 575, infer_only_fps=590), row(12, 580, infer_only_fps=590)]
    rec = recommend(rows)
    assert rec["best"]["workers"] == 12 and rec["knee"]["workers"] == 8
    assert any("GPU-bound" in n for n in rec["notes"])
    assert recommend([]) == {}


def test_render_has_speedup_and_failures():
    rows = [{"seg": 60, "decoder": "sequential", "workers": 0, "resize": False, "chunks": 3, "processed_fps": 200,
             "fps_min": 200, "fps_max": 200, "avg_cores_used": 1},
            row(4, 400, agreement={"identical": False, "precision_vs_ref": 0.999, "recall_vs_ref": 1.0}),
            {"seg": 10, "decoder": "threads", "workers": 8, "resize": True, "chunks": 18, "error": "boom"}]
    md = render_markdown(rows, "infer")
    assert "2.00" in md and "0.999/1.0" in md and "FAILED: boom" in md
