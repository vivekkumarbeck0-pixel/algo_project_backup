import win32gui
import win32api


class WindowDetector:

    # =================================
    # INIT
    # =================================

    def __init__(self):

        self.windows = []

    # =================================
    # Get All Visible Windows
    # =================================

    def enum(self):

        self.windows.clear()

        def callback(hwnd, extra):

            if win32gui.IsWindowVisible(hwnd):

                title = win32gui.GetWindowText(hwnd)

                if title:

                    left, top, right, bottom = (
                        win32gui.GetWindowRect(hwnd)
                    )

                    self.windows.append(
                        (
                            hwnd,
                            title,
                            left,
                            top,
                            right,
                            bottom
                        )
                    )

        win32gui.EnumWindows(
            callback,
            None
        )

        return self.windows

    # =================================
    # Find Window By Keyword
    # =================================

    def find(self, keyword):

        for (
            hwnd,
            title,
            left,
            top,
            right,
            bottom
        ) in self.enum():

            if keyword.lower() in title.lower():

                return (
                    hwnd,
                    title
                )

        return None

    # =================================
    # Bottom Left Window (CE)
    # =================================

    def find_bottom_left(self):

        windows = self.enum()

        if not windows:

            return None

        screen_w = win32api.GetSystemMetrics(0)
        screen_h = win32api.GetSystemMetrics(1)

        best = None
        score = -1

        for (
            hwnd,
            title,
            left,
            top,
            right,
            bottom
        ) in windows:

            cx = (
                left + right
            ) / 2

            cy = (
                top + bottom
            ) / 2

            # ---------------------------------
            # CE validation
            # ---------------------------------

            if (
                cx < screen_w / 2
                and cy > screen_h / 2
                and "C" in title
                and "NIFTY" in title.upper()
            ):

                if cy > score:

                    score = cy

                    best = (
                        hwnd,
                        title
                    )

        return best

    # =================================
    # Bottom Right Window (PE)
    # =================================

    def find_bottom_right(self):

        windows = self.enum()

        if not windows:

            return None

        screen_w = win32api.GetSystemMetrics(0)
        screen_h = win32api.GetSystemMetrics(1)

        best = None
        score = -1

        for (
            hwnd,
            title,
            left,
            top,
            right,
            bottom
        ) in windows:

            cx = (
                left + right
            ) / 2

            cy = (
                top + bottom
            ) / 2

            # ---------------------------------
            # PE validation
            # ---------------------------------

            if (
                cx > screen_w / 2
                and cy > screen_h / 2
                and "P" in title
                and "NIFTY" in title.upper()
            ):

                if cy > score:

                    score = cy

                    best = (
                        hwnd,
                        title
                    )

        return best