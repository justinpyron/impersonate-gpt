from dataclasses import dataclass


@dataclass
class ImpersonateConfig:
    name: str
    files_train: list[str]
    files_eval: list[str]


config_darwin = ImpersonateConfig(
    name="darwin",
    files_train=[
        "data/darwin-coral_reefs.txt",
        "data/darwin-descent_of_man.txt",
        "data/darwin-origin_of_species.txt",
        "data/darwin-voyage_of_the_beagle.txt",
    ],
    files_eval=["data/darwin-emotions_in_man_and_animals.txt"],
)

config_dostoevsky = ImpersonateConfig(
    name="dostoevsky",
    files_train=[
        "data/dostoevsky-brothers_karamazov.txt",
        "data/dostoevsky-crime_and_punishment.txt",
        "data/dostoevsky-notes_from_the_underground.txt",
        "data/dostoevsky-the_gambler.txt",
    ],
    files_eval=["data/dostoevsky-the_idiot.txt"],
)

config_fitzgerald = ImpersonateConfig(
    name="fitzgerald",
    files_train=[
        "data/fitzgerald-beautiful_and_damned.txt",
        "data/fitzgerald-flappers_and_philosophers.txt",
        "data/fitzgerald-great_gatsby.txt",
        "data/fitzgerald-tales_of_the_jazz_age.txt",
    ],
    files_eval=["data/fitzgerald-this_side_of_paradise.txt"],
)

config_twain = ImpersonateConfig(
    name="twain",
    files_train=[
        "data/twain-the_prince_and_the_pauper.txt",
        "data/twain-tom_sawyer.txt",
        "data/twain-huckleberry_finn.txt",
        "data/twain-life_on_the_mississippi.txt",
    ],
    files_eval=["data/twain-connecticut_yankee_king_arthurs_court.txt"],
)

configs = {
    "darwin": config_darwin,
    "dostoevsky": config_dostoevsky,
    "fitzgerald": config_fitzgerald,
    "twain": config_twain,
}
