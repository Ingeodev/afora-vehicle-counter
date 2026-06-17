from datetime import datetime, timedelta


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def calculate_exact_time(base_time: str, elapsed_seconds: float) -> str | None:
    try:
        parts = base_time.split(":")
        fmt = "%H:%M:%S" if len(parts) == 3 else "%H:%M"

        base_str = f"2000-01-01 {base_time}"
        base_dt = datetime.strptime(base_str, f"%Y-%m-%d {fmt}")
        result_dt = base_dt + timedelta(seconds=elapsed_seconds)

        return result_dt.strftime("%H:%M:%S")
    except ValueError:
        return None


def parse_time(time_str: str) -> tuple[int, int, int] | None:
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        elif len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        return None
    except ValueError:
        return None


def seconds_to_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
