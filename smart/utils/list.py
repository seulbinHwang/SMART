from typing import List, Sequence, TypeVar

T = TypeVar('T')


def safe_list_index(items: Sequence[T], value: T) -> int:
    """리스트에서 값을 찾고, 없으면 -1을 돌려준다.

    Args:
        items: 찾을 대상이 들어 있는 순서 자료.
        value: 찾고 싶은 값.

    Returns:
        값이 있으면 그 위치, 없으면 -1.
    """
    try:
        return list(items).index(value)
    except ValueError:
        return -1
