"""Verify the effective judge backend against its preserved source snapshot."""
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_backend(provenance, source, cfg):
    backend = provenance.get("backend", "transformers")
    assert backend in ("transformers", "vllm")
    module = "judge_vllm.py" if backend == "vllm" else "judge.py"
    assert provenance["source_sha256"] == digest(source / module)
    assert provenance["common_sha256"] == digest(source / "common.py")
    loaded = provenance["loaded_model"]
    assert loaded["model_class"] == "Qwen3_5ForConditionalGeneration"
    if backend == "transformers":
        assert set(loaded["parameter_dtypes"]) == {"torch.bfloat16"}
    assert loaded["generation"]["use_cache"]
    assert loaded["generation"]["max_new_tokens"] == cfg["max_new_tokens"]
    if backend == "vllm":
        runtime = json.loads((source / "judge-vllm.json").read_text())
        assert provenance["runtime"] == runtime
        assert provenance["runtime_sha256"] == digest(source / "judge-vllm.json")
        assert provenance["versions"]["vllm"] == runtime["version"]
        assert provenance["telemetry_enabled"] is True
        assert loaded["revision"] == cfg["revision"] and loaded["quantization"] is None
        assert loaded['resolved_revision'] == cfg['revision']
        assert loaded["visual_parameter_count"] > 0
        assert set(loaded['parameter_dtypes']) <= {'torch.bfloat16','torch.float32'}
        promoted = loaded['bf16_values_in_fp32_kernel_storage']
        assert all(name.endswith('.A_log') for name in promoted)
        assert len(promoted) == loaded['parameter_dtypes'].get('torch.float32',0)
        engine = provenance["engine_args"]
        assert all(engine[k] == v for k,v in runtime["engine"].items())
        assert engine["model"] == cfg["model_id"]
        assert engine["revision"] == engine["tokenizer_revision"] == cfg["revision"]
        sampling = provenance["sampling"]
        assert sampling["temperature"] == 0 and sampling["max_tokens"] == cfg["max_new_tokens"]
        assert sampling["repetition_penalty"] == 1 and sampling["ignore_eos"] is False
        assert sampling["presence_penalty"] == sampling["frequency_penalty"] == 0
