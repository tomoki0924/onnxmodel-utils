import weakref
from typing import Any, Optional


class WeakableList(list):
    pass


class WeakableDict(dict):
    pass


class WeakListEntity:
    def __init__(
        self,
        value: Any,
        prev_entity: Optional["WeakListEntity"] = None,
        next_entity: Optional["WeakListEntity"] = None,
    ):
        self._value = weakref.ref(value)
        self.prev = prev_entity
        self.next = next_entity

    @property
    def value(self) -> Optional[Any]:
        return self._value()

    def is_none(self) -> bool:
        return self._value() is None


class WeakList:
    _dummy_head_value = {"dummy_head"}
    _dummy_tail_value = {"dummy_tail"}

    def __init__(self):
        self._dummy_head = WeakListEntity(self._dummy_head_value)
        self._dummy_tail = WeakListEntity(self._dummy_tail_value)
        self._head = self._dummy_head
        self._tail = self._dummy_tail

    def next(self, entity: WeakListEntity) -> Optional[WeakListEntity]:
        while entity and entity != self._dummy_tail:
            entity = entity.next
            if entity is not None and not entity.is_none():
                return entity
        return self._dummy_tail

    def prev(self, entity: WeakListEntity) -> Optional[WeakListEntity]:
        while entity and entity != self._dummy_head:
            entity = entity.prev
            if entity is not None and not entity.is_none():
                return entity
        return self._dummy_head

    @property
    def head(self) -> Optional[WeakListEntity]:
        return self.next(self._dummy_head)

    @property
    def tail(self) -> Optional[WeakListEntity]:
        return self.prev(self._dummy_tail)

    @head.setter
    def head(self, entity: WeakListEntity) -> None:
        self._dummy_head.next = entity
        entity.prev = self._dummy_head
        self._head = entity

    @tail.setter
    def tail(self, entity: WeakListEntity) -> None:
        self._dummy_tail.prev = entity
        entity.next = self._dummy_tail
        self._tail = entity

    def append(self, value: Any) -> None:
        entity = WeakListEntity(value)
        if self.tail is None:
            self.head = entity
            self.tail = entity
        else:
            self.tail.next = entity
            entity.prev = self.tail
            self.tail = entity

    def appendleft(self, value: Any) -> None:
        entity = WeakListEntity(value)
        if self.head is None:
            self.head = entity
            self.tail = entity
        else:
            self.head.prev = entity
            entity.next = self.head
            self.head = entity

    def extend(self, other: "WeakList") -> None:
        for value in other:
            self.append(value)

    def __contains__(self, value: Any) -> bool:
        return value in list(self.__iter__())

    def __getitem__(self, index: int) -> Any:
        return list(self.__iter__())[index]

    def remove(self, entity: WeakListEntity) -> None:
        if entity == self._dummy_head or entity == self._dummy_tail:
            return
        if entity.prev is not None:
            entity.prev.next = entity.next
        if entity.next is not None:
            entity.next.prev = entity.prev
        if entity == self.head:
            self.head = entity.next
        if entity == self.tail:
            self.tail = entity.prev

    def remove_by_value(self, value: Any) -> None:
        entity = self.head
        while entity != self._dummy_tail:
            if entity.value == value:
                self.remove(entity)
            entity = self.next(entity)

    def __iter__(self):
        entity = self.head
        while entity != self._dummy_tail:
            if entity not in (self._dummy_head, self._dummy_tail):
                yield entity.value
            entity = self.next(entity)

    def __len__(self) -> int:
        return len(list(self.__iter__()))

    def __repr__(self) -> str:
        return f"WeakList({list(self.__iter__())})"

    def __str__(self) -> str:
        return self.__repr__()

    def __bool__(self) -> bool:
        return len(self) > 0

    @classmethod
    def from_list(cls, values) -> "WeakList":
        weak_list = cls()
        for value in values:
            weak_list.append(value)
        return weak_list


class AutoKeyDict:
    def __init__(self, value_to_key_func):
        self.v2k = value_to_key_func
        self._s = WeakList()
        self._d = dict()

    def remove(self, key: Any) -> None:
        if key in self._d:
            del self._d[key]

    def update(self) -> None:
        _d = dict()
        for v in self._s:
            _d[self.v2k(v)] = v
        self._d = _d

    def add(self, value: Any) -> None:
        key = self.v2k(value)
        if key in self._d:
            raise ValueError(f"Key {key} already exists")
        self._s.append(value)
        self._d[self.v2k(value)] = value

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self._d:
            raise ValueError(f"Key {key} already exists")
        self._d[key] = value
        self._s.append(value)

    def __getitem__(self, key: Any) -> Any:
        if key in self._d:
            return self._d[key]
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return key in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._s)

    def keys(self):
        return list(self.__iter__())

    def values(self):
        return list(self._d.values())

    def items(self):
        return list(self._d.items())

    def __eq__(self, other: Any) -> bool:
        return self._d == other._d

    def __str__(self) -> str:
        return str(self._d)
