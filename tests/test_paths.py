"""Rutas y perfil: los scripts apuntan a carpetas del proyecto y ningún módulo se salta el perfil elegido."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_training_data_folder_is_the_package_folder():
    """Los datos de entrenamiento viven dentro de training/."""
    from training import train_detector, train_recognizer
    assert train_recognizer.DIR == train_detector.DIR == ROOT / "training"


def test_only_the_profile_loader_names_a_profile():
    """El núcleo y los visores usan el perfil activo: importar uno concreto ignoraría VERIFICATION_PROFILE."""
    files = [*(ROOT / "verification").glob("*.py"), *(ROOT / "viewer").glob("*.py"),
             *(ROOT / "evaluation_viewer").glob("*.py")]
    assert not [f.name for f in files if "import" in f.read_text(encoding="utf-8")
                and "profiles.operator_certification" in f.read_text(encoding="utf-8")]
