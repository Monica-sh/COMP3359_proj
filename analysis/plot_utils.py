import pickle


class SaneDict(dict):
    # used in SacredUnpickler
    pass


class SaneList(list):
    # used in SacredUnpickler
    pass


def load_sacred_pickle(fp, **kwargs):
    """Unpickle an object that may contain Sacred ReadOnlyDict and ReadOnlyList
    objects. It will convert those objects to plain dicts/lists."""
    return SacredUnpickler(fp, **kwargs).load()


class SacredUnpickler(pickle.Unpickler):
    """Unpickler that replaces Sacred's ReadOnlyDict/ReadOnlyList with
    dict/list."""
    overrides = {
        # for some reason we need to replace dict with a custom class, or
        # else we get an AttributeError complaining that 'dict' has no
        # attribute '__dict__' (I don't know why this hapens)
        ('sacred.config.custom_containers', 'ReadOnlyDict'): SaneDict,
        ('sacred.config.custom_containers', 'ReadOnlyList'): SaneList,
    }

    def find_class(self, module, name):
        key = (module, name)
        if key in self.overrides:
            return self.overrides[key]
        return super().find_class(module, name)