import argparse
import os
import sys
from pathlib import Path

import cv2
import face_recognition
import numpy as np

# Import existing database sync & name parser utilities
from db import DB_FILE, KNOWN_FACES_DIR, get_db_connection, init_db, parse_clean_name

# ANSI terminal colors for readable CLI output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def diagnose_image(image_path: Path):
    """
    Performs comprehensive diagnostic checks on an individual photo:
      1. File readable check
      2. Image resolution & aspect ratio
      3. Mean brightness estimation
      4. Face detection (HOG / face_recognition)
      5. Face count and crop size validation
    """
    result = {
        "file": image_path.name,
        "clean_name": parse_clean_name(image_path.name),
        "status": "FAIL",
        "reason": "",
        "faces_detected": 0,
        "resolution": "N/A",
        "brightness": 0.0,
        "face_sizes": [],
        "encoding_shape": None,
    }

    # 1. Read file with OpenCV first to assess image matrix
    bgr_img = cv2.imread(str(image_path))
    if bgr_img is None:
        result["reason"] = "Corrupted image file or unsupported format"
        return result

    h, w, _ = bgr_img.shape
    result["resolution"] = f"{w}x{h}"

    # Calculate average grayscale brightness (0 = pitch black, 255 = washed out)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(np.mean(gray))
    result["brightness"] = avg_brightness

    # 2. Convert to RGB for face_recognition
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    try:
        # Detect face bounding boxes using default HOG model
        face_locations = face_recognition.face_locations(rgb_img)
        result["faces_detected"] = len(face_locations)

        if len(face_locations) == 0:
            notes = []
            if avg_brightness < 40:
                notes.append("Lighting too dark")
            elif avg_brightness > 220:
                notes.append("Overexposed/blown out")
            if min(w, h) < 150:
                notes.append("Resolution too low")

            reason = "No face found by detector"
            if notes:
                reason += f" ({', '.join(notes)})"
            else:
                reason += " (Check head tilt, side profile, or occlusion)"
            result["reason"] = reason
            return result

        # Record bounding box dimensions for all detected faces
        for top, right, bottom, left in face_locations:
            fw, fh = right - left, bottom - top
            result["face_sizes"].append(f"{fw}x{fh}")

        # Warn if multiple individuals appear in an enrollment headshot
        if len(face_locations) > 1:
            result["status"] = "WARN"
            result["reason"] = (
                f"Multiple faces detected ({len(face_locations)}). "
                "Primary face will be used, but single-person headshots are recommended."
            )
        else:
            fw, fh = (
                face_locations[0][1] - face_locations[0][3],
                face_locations[0][2] - face_locations[0][0],
            )
            if fw < 70 or fh < 70:
                result["status"] = "WARN"
                result["reason"] = (
                    f"Face crop is very small ({fw}x{fh}px). Recognition may be unreliable."
                )
            else:
                result["status"] = "PASS"

        # 3. Verify embedding generation
        encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if len(encodings) > 0:
            result["encoding_shape"] = encodings[0].shape
        else:
            result["status"] = "FAIL"
            result["reason"] = (
                "Face detected, but facial landmarks could not be aligned."
            )

    except Exception as e:
        result["status"] = "FAIL"
        result["reason"] = f"Extraction error: {str(e)}"

    return result


def inspect_sqlite_db():
    """Reads existing SQLite DB statistics to check current enrollment numbers."""
    if not os.path.exists(DB_FILE):
        return None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total_users FROM users")
        total_users = cursor.fetchone()["total_users"]

        cursor.execute("SELECT COUNT(*) AS total_vectors FROM face_embeddings")
        total_vectors = cursor.fetchone()["total_vectors"]

        cursor.execute("""
            SELECT u.name, COUNT(e.id) as photo_count
            FROM users u
            LEFT JOIN face_embeddings e ON u.id = e.user_id
            GROUP BY u.id
            ORDER BY u.name ASC
        """)
        user_breakdown = cursor.fetchall()
        conn.close()

        return {
            "users": total_users,
            "vectors": total_vectors,
            "breakdown": user_breakdown,
        }
    except Exception:
        return None


def run_diagnostics(target_dir: str, save_to_db: bool = False):
    folder = Path(target_dir)

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}🔍 FACE RECOGNITION DATASET DIAGNOSTIC TOOL{RESET}")
    print(f"{BOLD}Target Folder:{RESET} {folder.resolve()}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    if not folder.exists():
        print(f"{RED}❌ Error: Folder '{folder}' does not exist.{RESET}")
        print(f"Creating empty folder at {folder}...")
        folder.mkdir(parents=True, exist_ok=True)
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in valid_extensions]
    )

    if not files:
        print(
            f"{YELLOW}⚠️  No images found in '{folder}'. Supported formats: {', '.join(valid_extensions)}{RESET}"
        )
        return

    print(
        f"Discovered {BOLD}{len(files)}{RESET} candidate photos. Running diagnostics...\n"
    )

    passed_count = 0
    warn_count = 0
    fail_count = 0
    person_groups = {}

    # Header for report table
    print(
        f"{BOLD}{'STATUS':<8} | {'FILE NAME':<24} | {'PARSED USER':<14} | {'FACE':<7} | {'DETAILS'}{RESET}"
    )
    print("-" * 75)

    for file_path in files:
        diag = diagnose_image(file_path)
        status = diag["status"]
        user_name = diag["clean_name"]

        # Track per-person photo counts
        if user_name not in person_groups:
            person_groups[user_name] = {"pass": 0, "fail": 0}

        if status == "PASS":
            passed_count += 1
            person_groups[user_name]["pass"] += 1
            status_badge = f"{GREEN}[PASS]{RESET}"
            details = f"Dim: {diag['resolution']}, Face: {diag['face_sizes'][0]}"
        elif status == "WARN":
            warn_count += 1
            person_groups[user_name]["pass"] += 1
            status_badge = f"{YELLOW}[WARN]{RESET}"
            details = diag["reason"]
        else:
            fail_count += 1
            person_groups[user_name]["fail"] += 1
            status_badge = f"{RED}[FAIL]{RESET}"
            details = diag["reason"]

        face_count_str = f"{diag['faces_detected']} face"
        print(
            f"{status_badge:<17} | {diag['file']:<24} | {user_name:<14} | {face_count_str:<7} | {details}"
        )

    # Summary section
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}📊 DIAGNOSTIC SUMMARY{RESET}")
    print(f"{'='*70}")
    print(f"Total Images Evaluated : {len(files)}")
    print(f"{GREEN}✓ Passed (Ready for DB): {passed_count}{RESET}")
    print(f"{YELLOW}▲ Warnings (Inspect)   : {warn_count}{RESET}")
    print(f"{RED}✗ Failed (Undetectable): {fail_count}{RESET}")

    # Enrollment coverage by person
    print(f"\n{BOLD}👥 Verified Sample Coverage per Person:{RESET}")
    for person, counts in sorted(person_groups.items()):
        total = counts["pass"] + counts["fail"]
        color = (
            GREEN if counts["pass"] >= 3 else (YELLOW if counts["pass"] > 0 else RED)
        )
        note = " (Ideal: 3-5 photos)" if counts["pass"] < 3 else ""
        print(
            f"  - {BOLD}{person:<16}{RESET} : {color}{counts['pass']}/{total} detectable{RESET}{note}"
        )

    # SQLite Database Status
    db_stats = inspect_sqlite_db()
    if db_stats:
        print(f"\n{BOLD}🗄️  Current SQLite Database State ('{DB_FILE}'):{RESET}")
        print(f"  - Enrolled Unique Users   : {db_stats['users']}")
        print(f"  - Total Face Vectors in DB: {db_stats['vectors']}")
        for row in db_stats["breakdown"]:
            print(f"      • {row['name']}: {row['photo_count']} indexed vectors")

    # Optional auto-sync trigger
    if save_to_db:
        from db import scan_and_sync_folder

        print(
            f"\n{BOLD}{CYAN}🔄 Syncing detectable photos directly into SQLite...{RESET}"
        )
        report = scan_and_sync_folder(folder_path=str(folder))
        print(
            f"{GREEN}Sync finished: +{report['loaded']} added, {report['skipped']} already present, {len(report['failed'])} rejected.{RESET}"
        )
    else:
        print(
            f"\n{BOLD}💡 Tip:{RESET} Pass {CYAN}--sync{RESET} to automatically register all detectable images into SQLite:"
        )
        print(f"   {BOLD}python diagnose_faces.py --sync{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose photos in known_faces folder before running Streamlit."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=KNOWN_FACES_DIR,
        help="Path to folder containing known face photos (default: ./known_faces)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Automatically sync verified photos into faces.db after diagnostics.",
    )

    args = parser.parse_args()
    run_diagnostics(target_dir=args.folder, save_to_db=args.sync)
