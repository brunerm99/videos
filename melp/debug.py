from manimlib import Integer, Mobject, VGroup

__all__ = ["index_labels"]


def index_labels(mobject: Mobject, label_height: float = 0.15, **kwargs):
    labels = VGroup()

    for index, m in enumerate(mobject):
        label = Integer(index, **kwargs)
        label.set_height(label_height)
        label.move_to(m)
        labels.add(label)

    return labels
