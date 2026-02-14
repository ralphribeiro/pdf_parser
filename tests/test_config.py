"""
Testes TDD para config.py refatorado com python-dotenv.

Escritos ANTES da implementação para guiar o desenvolvimento.
Cada teste valida que config.py lê variáveis de ambiente com prefixo
DOC_PARSER_ e mantém backward-compat com a interface de módulo.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _reload_config():
    """
    Recarrega o módulo config para aplicar novas env vars.

    Necessário porque Python cacheia módulos em sys.modules.
    Mock de load_dotenv impede que o arquivo .env local
    interfira nos testes (só env vars explícitas contam).
    """
    if "config" in sys.modules:
        del sys.modules["config"]

    import dotenv

    with patch.object(dotenv, "load_dotenv", return_value=False):
        import config

    return config


# =========================================================================
# Defaults (sem env vars definidas)
# =========================================================================


class TestConfigDefaults:
    """Valores default quando nenhuma DOC_PARSER_* env var está definida."""

    def test_use_gpu_is_bool(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_USE_GPU", raising=False)
        config = _reload_config()
        assert isinstance(config.USE_GPU, bool)

    def test_device_cpu_when_gpu_false(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_USE_GPU", "false")
        config = _reload_config()
        assert config.DEVICE == "cpu"
        assert config.USE_GPU is False

    def test_device_cuda_when_gpu_true(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_USE_GPU", "true")
        config = _reload_config()
        assert config.DEVICE == "cuda"
        assert config.USE_GPU is True

    def test_ocr_engine_default(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_OCR_ENGINE", raising=False)
        config = _reload_config()
        assert config.OCR_ENGINE == "doctr"

    def test_ocr_dpi_default(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_OCR_DPI", raising=False)
        config = _reload_config()
        assert config.OCR_DPI == 350
        assert isinstance(config.OCR_DPI, int)

    def test_min_confidence_default(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_MIN_CONFIDENCE", raising=False)
        config = _reload_config()
        assert config.MIN_CONFIDENCE == 0.3
        assert isinstance(config.MIN_CONFIDENCE, float)

    def test_ocr_batch_size_default(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_OCR_BATCH_SIZE", raising=False)
        config = _reload_config()
        assert config.OCR_BATCH_SIZE == 20

    def test_verbose_default_true(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_VERBOSE", raising=False)
        config = _reload_config()
        assert config.VERBOSE is True

    def test_parallel_enabled_default_true(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_PARALLEL_ENABLED", raising=False)
        config = _reload_config()
        assert config.PARALLEL_ENABLED is True

    def test_parallel_workers_default_none(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_PARALLEL_WORKERS", raising=False)
        config = _reload_config()
        assert config.PARALLEL_WORKERS is None

    def test_ocr_postprocess_default_true(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_OCR_POSTPROCESS", raising=False)
        config = _reload_config()
        assert config.OCR_POSTPROCESS is True

    def test_ocr_fix_errors_default_true(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_OCR_FIX_ERRORS", raising=False)
        config = _reload_config()
        assert config.OCR_FIX_ERRORS is True

    def test_ocr_lang_default_por(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_OCR_LANG", raising=False)
        config = _reload_config()
        assert config.OCR_LANG == "por"

    def test_assume_straight_pages_default_true(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_ASSUME_STRAIGHT_PAGES", raising=False)
        config = _reload_config()
        assert config.ASSUME_STRAIGHT_PAGES is True

    def test_detect_orientation_default_false(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_DETECT_ORIENTATION", raising=False)
        config = _reload_config()
        assert config.DETECT_ORIENTATION is False

    def test_searchable_pdf_default_true(self, monkeypatch):
        monkeypatch.delenv("DOC_PARSER_SEARCHABLE_PDF", raising=False)
        config = _reload_config()
        assert config.SEARCHABLE_PDF is True


# =========================================================================
# Env var overrides
# =========================================================================


class TestConfigEnvOverrides:
    """Cada DOC_PARSER_* env var sobrescreve o default correspondente."""

    def test_ocr_engine_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_OCR_ENGINE", "tesseract")
        config = _reload_config()
        assert config.OCR_ENGINE == "tesseract"

    def test_ocr_dpi_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_OCR_DPI", "300")
        config = _reload_config()
        assert config.OCR_DPI == 300

    def test_min_confidence_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_MIN_CONFIDENCE", "0.5")
        config = _reload_config()
        assert config.MIN_CONFIDENCE == 0.5

    def test_ocr_batch_size_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_OCR_BATCH_SIZE", "8")
        config = _reload_config()
        assert config.OCR_BATCH_SIZE == 8

    def test_verbose_false_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_VERBOSE", "false")
        config = _reload_config()
        assert config.VERBOSE is False

    def test_parallel_enabled_false_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_PARALLEL_ENABLED", "false")
        config = _reload_config()
        assert config.PARALLEL_ENABLED is False

    def test_parallel_workers_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_PARALLEL_WORKERS", "4")
        config = _reload_config()
        assert config.PARALLEL_WORKERS == 4

    def test_ocr_postprocess_false_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_OCR_POSTPROCESS", "false")
        config = _reload_config()
        assert config.OCR_POSTPROCESS is False

    def test_ocr_fix_errors_false_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_OCR_FIX_ERRORS", "false")
        config = _reload_config()
        assert config.OCR_FIX_ERRORS is False

    def test_ocr_lang_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_OCR_LANG", "por+eng")
        config = _reload_config()
        assert config.OCR_LANG == "por+eng"

    def test_assume_straight_pages_false_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_ASSUME_STRAIGHT_PAGES", "false")
        config = _reload_config()
        assert config.ASSUME_STRAIGHT_PAGES is False

    def test_detect_orientation_true_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_DETECT_ORIENTATION", "true")
        config = _reload_config()
        assert config.DETECT_ORIENTATION is True

    def test_straighten_pages_true_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_STRAIGHTEN_PAGES", "true")
        config = _reload_config()
        assert config.STRAIGHTEN_PAGES is True

    def test_output_dir_from_env(self, monkeypatch, tmp_path):
        custom_dir = str(tmp_path / "custom_output")
        monkeypatch.setenv("DOC_PARSER_OUTPUT_DIR", custom_dir)
        config = _reload_config()
        assert config.OUTPUT_DIR == Path(custom_dir)

    def test_searchable_pdf_false_from_env(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_SEARCHABLE_PDF", "false")
        config = _reload_config()
        assert config.SEARCHABLE_PDF is False


# =========================================================================
# Bool parsing edge cases
# =========================================================================


class TestConfigBoolParsing:
    """Valores truthy/falsy aceitos para env vars booleanas."""

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "Yes"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("DOC_PARSER_USE_GPU", value)
        config = _reload_config()
        assert config.USE_GPU is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "No"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("DOC_PARSER_USE_GPU", value)
        config = _reload_config()
        assert config.USE_GPU is False


# =========================================================================
# Backward compatibility
# =========================================================================


class TestConfigBackwardCompat:
    """Todas as variáveis de módulo originais devem continuar existindo."""

    def test_all_expected_attributes_exist(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_USE_GPU", "false")
        config = _reload_config()

        expected = [
            # Diretórios
            "BASE_DIR", "RESOURCE_DIR", "OUTPUT_DIR",
            # Device
            "DEVICE", "USE_GPU",
            # OCR geral
            "OCR_ENGINE", "OCR_DPI", "IMAGE_DPI", "MIN_CONFIDENCE",
            "OCR_BATCH_SIZE",
            # OCR orientação
            "ASSUME_STRAIGHT_PAGES", "DETECT_ORIENTATION", "STRAIGHTEN_PAGES",
            # OCR tesseract
            "OCR_LANG", "TESSERACT_CONFIG",
            # OCR pós-processamento
            "OCR_POSTPROCESS", "OCR_FIX_ERRORS", "OCR_MIN_LINE_LENGTH",
            # Pré-processamento (legado)
            "OCR_PREPROCESS", "BINARIZATION_METHOD", "DENOISE_KERNEL_SIZE",
            "DESKEW_ANGLE_THRESHOLD",
            # Tabelas
            "TABLE_DETECTION_CONFIDENCE", "CAMELOT_FLAVOR",
            # Detecção de página
            "IMAGE_AREA_THRESHOLD", "TEXT_COVERAGE_THRESHOLD",
            # PDF pesquisável
            "SEARCHABLE_PDF",
            # Paralelização
            "PARALLEL_ENABLED", "PARALLEL_WORKERS", "PARALLEL_MIN_PAGES",
            # Debug
            "SAVE_PREPROCESSED_IMAGES", "VERBOSE",
        ]

        for attr in expected:
            assert hasattr(config, attr), f"config.{attr} deve existir"

    def test_base_dir_is_project_root(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_USE_GPU", "false")
        config = _reload_config()
        assert config.BASE_DIR.is_dir()
        assert (config.BASE_DIR / "config.py").exists()

    def test_resource_dir_relative_to_base(self, monkeypatch):
        monkeypatch.setenv("DOC_PARSER_USE_GPU", "false")
        config = _reload_config()
        assert config.RESOURCE_DIR == config.BASE_DIR / "resource"

    def test_image_dpi_equals_ocr_dpi(self, monkeypatch):
        """IMAGE_DPI deve ser alias de OCR_DPI."""
        monkeypatch.setenv("DOC_PARSER_OCR_DPI", "200")
        config = _reload_config()
        assert config.IMAGE_DPI == config.OCR_DPI == 200
