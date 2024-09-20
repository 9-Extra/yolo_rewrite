from collections.abc import Callable
from typing import Optional, Literal

TargetFunction = Callable[[], None]


class _Context:
    state: dict[Literal["running", "completed"], set[str]]
    targets: dict[TargetFunction, "Target"]

    def __init__(self):
        self.targets = {}
        self.state = {"running": set(), "completed": set()}

    def register_target(self, func: TargetFunction, target: "Target"):
        assert target.name not in self.targets
        self.targets[func] = target

    def on_target_start(self, target: "Target"):
        self.state["running"].add(target.name)

    def on_target_complete(self, target: "Target"):
        self.state["running"].remove(target.name)
        self.state["completed"].add(target.name)


_context = _Context()


class Target:
    name: str
    dependence: tuple[TargetFunction, ...]

    def __init__(self, *dependence: TargetFunction, name: Optional[str] = None):
        self.dependence = dependence
        self.name = name # type: ignore

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
        target_reg = _context.targets[func]
        if target_reg.name not in _context.state["completed"]:
            _context.on_target_start(target_reg)
            print(f"Running Target: {target_reg.name}")
            func()
            _context.on_target_complete(target_reg)


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


def get_target_by_name(name: str) -> Optional[TargetFunction]:
    global _context
    for func, target in _context.targets.items():
        if target.name == name or target.name == "_" + name:
            return func

    return None

def get_target_names() -> list[str]:
    global _context
    return [target.name.removeprefix("_") for target in _context.targets.values()]