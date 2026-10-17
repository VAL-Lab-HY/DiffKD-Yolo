import importlib

# name → (module, class). Imported lazily so each teacher only needs its own dependencies.
TEACHERS = {
    "retinexformer": ("models.teachers.retinexformer", "RetinexFormer"),
    "irformer": ("models.teachers.irformer", "Model"),
    "mbllen": ("models.teachers.mbllen", "MBLLEN"),
}


def build_teacher(name, **kwargs):
    if name not in TEACHERS:
        raise KeyError(f"Unknown teacher '{name}'. Available: {list(TEACHERS)}")
    module_name, class_name = TEACHERS[name]
    return getattr(importlib.import_module(module_name), class_name)(**kwargs)
