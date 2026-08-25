from readers.window_detector import WindowDetector


detector = WindowDetector()

windows = detector.enum()

print()
print("=" * 80)
print("VISIBLE WINDOWS")
print("=" * 80)

for hwnd, title, left, top, right, bottom in windows:

    print()
    print("TITLE :", title)
    print("HWND  :", hwnd)
    print("POS   :", left, top, right, bottom)

print()
print("=" * 80)
print("BOTTOM LEFT")
print("=" * 80)

print(
    detector.find_bottom_left()
)

print()
print("=" * 80)
print("BOTTOM RIGHT")
print("=" * 80)

print(
    detector.find_bottom_right()
)

print()