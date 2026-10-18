"""Download the OPSIN command-line JAR used by the extraction notebook."""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


VERSION = "2.8.0"
FILENAME = f"opsin-cli-{VERSION}-jar-with-dependencies.jar"
DOWNLOAD_URL = (
    f"https://github.com/dan2097/opsin/releases/download/{VERSION}/{FILENAME}"
)
DESTINATION = Path(__file__).resolve().parent / FILENAME


def main() -> None:
    if DESTINATION.exists():
        print(f"OPSIN already exists: {DESTINATION}")
        return

    temporary_file = DESTINATION.with_suffix(f"{DESTINATION.suffix}.part")
    print(f"Downloading OPSIN {VERSION} from the official GitHub release...")

    try:
        urlretrieve(DOWNLOAD_URL, temporary_file)
        temporary_file.replace(DESTINATION)
    except (HTTPError, URLError, OSError) as error:
        temporary_file.unlink(missing_ok=True)
        raise SystemExit(f"Failed to download OPSIN: {error}") from error

    print(f"Downloaded: {DESTINATION}")


if __name__ == "__main__":
    main()
