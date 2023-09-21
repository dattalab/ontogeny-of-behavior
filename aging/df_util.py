from toolz import curry


@curry
def get_index(df, level):
    return df.index.get_level_values(level)

get_age = get_index(level='age')
get_mouse = get_index(level='mouse_id')
