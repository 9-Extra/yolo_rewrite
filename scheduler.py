from collections.abc import Callable
from typing import Optional

TargetFunction = Callable[[], None]
TargetIndex = TargetFunction | str # 可以使用名称或者函数本身找到目标对象


class _Context:
    completed: set["Target"]
    targets: dict[str, "Target"]
    targets_from_func: dict[TargetFunction, "Target"]

    def __init__(self):
        self.targets = {}
        self.targets_from_func = {}
        self.completed = set()

    def register_target(self, target: "Target"):
        assert target.name not in self.targets, f"{target.name}重复出现"
        self.targets[target.name] = target
        self.targets_from_func[target.func] = target

    def on_target_start(self, target: "Target"):
        pass

    def on_target_complete(self, target: "Target"):
        self.completed.add(target.name)
        
    def get_target(self, target_idx: TargetIndex):
        if isinstance(target_idx, Callable):
            return self.targets_from_func[target_idx]
        elif isinstance(target_idx, str):
            return self.targets[target_idx]
        else:
            raise TypeError(f"只能使用函数或者名称")


_context = _Context()


class Target:
    func: TargetFunction
    name: str
    dependence: tuple[TargetIndex, ...]

    def __init__(self, *dependence: TargetIndex, name: Optional[str] = None):
        self.dependence = dependence # type: ignore
        self.name = name # type: ignore
        self.func = None # type: ignore

    def __call__(self, func: TargetFunction):
        if self.name is None:
            self.name = func.__name__

        self.func = func
        global _context
        _context.register_target(self)

        return func


def _dependency_solve(target_idx: TargetIndex) -> list[Target]:
    global _context
    visit_list: list[Target] = []
    to_visit = [target_idx]

    while len(to_visit) != 0:
        to_visit_next = []
        for t_id in to_visit:
            t = _context.get_target(t_id)
            if t not in visit_list:
                visit_list.append(t)
                to_visit_next.extend(t.dependence)
        to_visit = to_visit_next

    visit_list.reverse()  # 执行顺序与遍历顺序相反
    return visit_list


def run_target(target_idx: TargetIndex):
    global _context
    execute = _dependency_solve(target_idx)

    for target in execute:
        if target not in _context.completed:
            _context.on_target_start(target)
            print(f"Running Target: {target.name}")
            target.func() # run
            _context.on_target_complete(target)


def re_run_target(target_idx: TargetIndex):
    global _context
    execute = _dependency_solve(target_idx)
    cur_target = _context.get_target(target_idx)

    for target in execute:
        if target not in _context.completed or target is cur_target:
            _context.on_target_start(target)
            print(f"Running Target: {target.name}")
            target.func() # run
            _context.on_target_complete(target)
            
    


def virtual_run_target(target_idx: TargetIndex):
    global _context
    execute = _dependency_solve(target_idx)

    for target in execute:
        if target not in _context.completed:
            print(f"Running Target: {target.name}")
        else:
            print(f"Skip Target: {target.name}")


def get_target_by_name(name: str) -> Optional[Target]:
    global _context
    for n, target in _context.targets.items():
        if n == name or n == "_" + name:
            return target

    return None

def get_target_names() -> list[str]:
    global _context
    return [name.removeprefix("_") for name in _context.targets.keys()]