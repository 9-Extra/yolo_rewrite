import os
import atexit
from collections.abc import Callable
from typing import Optional, Literal

TargetFunction = Callable[[], None]


class _Context:
    state: dict[Literal["running", "completed"], set[str]]
    targets: dict[TargetFunction, "Target"]
    file_state_record: Optional[str]

    def __init__(self):
        self.targets = {}
        self.state = {"running": set(), "completed": set()}
        atexit.register(self._dump_state)

    def init_state(self, file_state_record):
        os.makedirs(os.path.dirname(file_state_record), exist_ok=True)
        self.file_state_record = file_state_record
        if os.path.isfile(file_state_record):
            self.state = eval(open(file_state_record, "r").read())

    def register_target(self, func: TargetFunction, target: "Target"):
        assert target.name not in self.targets
        self.targets[func] = target

    def _dump_state(self):
        if self.file_state_record is not None:
            open(self.file_state_record, "w").write(repr(self.state))

    def on_target_start(self, target: "Target"):
        self.state["running"].add(target.name)
        self._dump_state()

    def on_target_complete(self, target: "Target"):
        self.state["running"].remove(target.name)
        self.state["completed"].add(target.name)
        self._dump_state()


_context = _Context()


def init_context(file_state_record: str):
    global _context
    _context.init_state(file_state_record)


class Target:
    name: str
    dependence: tuple[TargetFunction, ...]

    def __init__(self, *dependence: TargetFunction, name: Optional[str] = None):
        self.dependence = dependence
        self.name = name

    def __call__(self, func: TargetFunction):
        if self.name is None:
            self.name = func.__name__

        global _context
        _context.register_target(func, self)

        return func


def _dependency_solve(target: TargetFunction) -> list[TargetFunction]:
    global _context
    visit_list = []
    visited = set()
    to_visit = [target]

    while len(to_visit) != 0:
        to_visit_next = []
        for target in to_visit:
            if target not in visited:
                visited.add(target)
                visit_list.append(target)
                to_visit_next.extend(_context.targets[target].dependence)
        to_visit = to_visit_next

    visit_list.reverse()  # 执行顺序与遍历顺序相反
    return visit_list


def run_target(target: TargetFunction):
    global _context
    execute = _dependency_solve(target)

    for func in execute:
        target = _context.targets[func]
        if target.name not in _context.state["completed"]:
            _context.on_target_start(target)
            print(f"Running Target: {target.name}")
            func()
            _context.on_target_complete(target)


def re_run_target(target: TargetFunction):
    global _context
    execute = _dependency_solve(target)

    for func in execute:
        target_info = _context.targets[func]
        if target_info.name not in _context.state["completed"] or func is target:
            _context.on_target_start(target_info)
            print(f"Running Target: {target_info.name}")
            func()
            _context.on_target_complete(target_info)


def virtual_run_target(target: TargetFunction):
    global _context
    execute = _dependency_solve(target)

    for func in execute:
        print(f"Run {_context.targets[func].name}")
