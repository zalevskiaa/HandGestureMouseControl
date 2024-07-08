def compute_screen_coords(hand_x, hand_y, rect_lt, rect_rb):

    rect_width = rect_rb[0] - rect_lt[0]
    rect_height = rect_rb[1] - rect_lt[1]

    padding = 0.001
    if hand_x < rect_lt[0]:
        hand_x = rect_lt[0]
    elif hand_x > rect_rb[0]:
        hand_x = rect_rb[0]

    if hand_y < rect_lt[1]:
        hand_y = rect_lt[1]
    elif hand_y > rect_rb[1]:
        hand_y = rect_rb[1]

    screen_width, screen_height = 1920, 1080
    screen_x = int((hand_x - rect_lt[0]) * screen_width / rect_width)
    screen_y = int((hand_y - rect_lt[1]) * screen_height / rect_height)

    return screen_x, screen_y
