from pathlib import Path

import kagglehub


def main() -> None:
    cache_path = Path(kagglehub.dataset_download("robertnowak/bowel-sounds"))
    print(cache_path / "data")


if __name__ == "__main__":
    main()
