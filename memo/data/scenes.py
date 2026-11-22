"""The scenes of the three datasets MEMO is evaluated on."""

SCENES = {
    "blender_moments": ["alligator", "beetle", "bicycle", "book", "elephant", "ficus_4d",
                        "lego_4d", "spiderman", "toyota"],
    "n3d": ["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1",
            "flame_steak", "sear_steak"],
    "selfcap": ["hair1", "hair2", "corgi1", "corgi2", "bike1", "bike2", "yoga"],
}

# Synthetic scenes are composited over white, captured ones over black.
WHITE_BACKGROUND = {"blender_moments": True, "n3d": False, "selfcap": False}


def select(datasets=None, scenes=None):
    """[(dataset, scene)] for the given datasets and scenes, everything by
    default. Raises ValueError naming the valid names on a typo, or when a
    scene is not in the selected datasets."""
    all_scenes = {s: d for d, ss in SCENES.items() for s in ss}
    unknown = [d for d in datasets or [] if d not in SCENES]
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; valid: {', '.join(SCENES)}")
    unknown = [s for s in scenes or [] if s not in all_scenes]
    if unknown:
        raise ValueError(f"unknown scenes {unknown}; valid: {', '.join(all_scenes)}")
    chosen = [(d, s) for d, ss in SCENES.items() for s in ss
              if (not datasets or d in datasets) and (not scenes or s in scenes)]
    outside = [s for s in scenes or [] if (all_scenes[s], s) not in chosen]
    if outside:
        raise ValueError(f"scenes {outside} are not in the datasets {datasets}")
    return chosen
