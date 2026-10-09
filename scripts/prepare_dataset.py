"""Compatibility entry point; install the package with `pip install -e .` first."""
from road_damage.preparation import convert_annotation as convert_annotation
from road_damage.preparation import main
from road_damage.preparation import prepare as prepare

if __name__ == "__main__":
    main()
