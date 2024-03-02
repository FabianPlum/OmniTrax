from . import ui, operators


def register():
    ui.register()
    operators.register()


def unregister():
    ui.unregister()
    operators.unregister()
