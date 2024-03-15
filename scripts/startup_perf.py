import time
from contextlib import ContextDecorator
from itertools import product
from pathlib import Path
from typing import Any

import fastapi
import pandas as pd
import pydantic
from fastapi.testclient import TestClient

assert pydantic.version.VERSION.startswith("2."), "Expect pydantic v2 only."



class timer(ContextDecorator):
    def __init__(self, msg: str = "", silent: bool = False):
        self.msg = msg
        self.elapsed = 0.0
        self.silent = silent

    def __enter__(self) -> Any:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> Any:
        self.elapsed = time.perf_counter() - self.start_time
        if not self.silent:
            print(f"{self.elapsed * 1000:.1f} ms\t\t{self.msg} ")


def build_model(
    n_fields_per_nesting_level: int,
    n_nesting_levels: int,
) -> type[pydantic.BaseModel]:
    fields: dict[str, tuple[type, Any]] = {
        f"f{i}": (str, f"x{i}") for i in range(n_fields_per_nesting_level)
    }

    _A = pydantic.create_model(f"_A", **fields)  # type: ignore
    _B = pydantic.create_model(f"_B", **fields)  # type: ignore
    _C = pydantic.create_model(f"_C", **fields)  # type: ignore
    _D = pydantic.create_model(f"_C", **fields)  # type: ignore

    class A(_A):
        pass

    class B(_B):
        sub_model: A = pydantic.Field(default_factory=A)

    class C(_C):
        sub_mode: B = pydantic.Field(default_factory=B)

    class D(_D):
        sub_model: C = pydantic.Field(default_factory=C)

    return_map = {0: A, 1: B, 2: C, 3: D}
    if n_nesting_levels not in return_map.keys():
        raise ValueError(
            f"Only {list(return_map.keys())} nesting levels are supported."
        )
    return return_map[n_nesting_levels]


def build_app(
    app: fastapi.FastAPI,
    n_sub_routers: int,
    n_routes: int,
    response_models: list[type[pydantic.BaseModel]],
) -> None:
    if n_sub_routers > 0:
        attachable: fastapi.APIRouter | fastapi.FastAPI = fastapi.APIRouter()
    else:
        attachable = app
    for i in range(n_routes):
        _response_model = response_models[i]

        @attachable.get(f"/{i}", response_model=_response_model)
        def index() -> Any:
            return {}  # type: ignore

    if n_sub_routers > 0:
        assert isinstance(attachable, fastapi.APIRouter)
        highest_level_router = attachable
        for _ in range(n_sub_routers - 1):
            new_router = fastapi.APIRouter()
            new_router.include_router(highest_level_router)
            highest_level_router = new_router
        app.include_router(highest_level_router)


def time_app(
    n_sub_routers: int,
    n_routes: int,
    n_fields_per_nesting_level: int,
    n_nesting_levels: int,
    use_unique_models: bool,
    silent=False,
) -> dict:
    if use_unique_models:
        response_models = [
            build_model(
                n_fields_per_nesting_level=n_fields_per_nesting_level,
                n_nesting_levels=n_nesting_levels,
            )
            for _ in range(n_routes)
        ]
        assert id(response_models[0]) != id(response_models[1])
    else:
        response_models = [
            build_model(
                n_fields_per_nesting_level=n_fields_per_nesting_level,
                n_nesting_levels=n_nesting_levels,
            )
        ] * n_routes

    with timer("instantiate app", silent=silent):
        app = fastapi.FastAPI()
    with timer(f"define routes", silent=silent) as build_app_t:
        build_app(
            app,
            n_sub_routers=n_sub_routers,
            n_routes=n_routes,
            response_models=response_models,
        )
    with timer("instantiate test client", silent=silent):
        testclient = TestClient(app)
    with timer(f"test client get 1", silent=silent) as test_client_get_1_t:
        assert "f0" in testclient.get(f"/0").json().keys()
    with timer(f"test client get 2", silent=silent) as test_client_get_2_t:
        assert "f0" in testclient.get(f"/0").json().keys()
    return {
        "build_app": build_app_t.elapsed,
        "test_client_get_1": test_client_get_1_t.elapsed,
        "test_client_get_2": test_client_get_2_t.elapsed,
    }


def main() -> None:
    n_sub_routers = 1  # Number of sub routers that the routes are added to the app via.
    n_routes = 300  # Number of routes in the entire app.
    n_fields_per_nesting_level = 10  # Fields per model, same at each nesting level.
    n_nesting_levels = 0  # Levels of nesting in pydantic models.
    use_unique_models = True  # Whether to use unique models for each route, or the same model for each route.
    time_app(
        n_sub_routers=n_sub_routers,
        n_routes=n_routes,
        n_fields_per_nesting_level=n_fields_per_nesting_level,
        n_nesting_levels=n_nesting_levels,
        use_unique_models=use_unique_models,
    )

    all_n_sub_routers = [0, 1, 2]
    all_n_routes = [2, 5, 10, 15, 30, 60]
    all_n_fields_per_nesting_level = [1, 5, 10, 15]
    all_n_nesting_levels = [0, 1, 2, 3]
    all_use_unique_models = [True, False]
    args = list(
        product(
            all_n_sub_routers,
            all_n_routes,
            all_n_fields_per_nesting_level,
            all_n_nesting_levels,
            all_use_unique_models,
        )
    )
    print(f"Running {len(args)} tests.")
    all_data = []
    for i, (
        n_sub_routers,
        n_routes,
        n_fields_per_nesting_level,
        n_nesting_levels,
        use_unique_models,
    ) in enumerate(args):
        print(f"Test {i + 1}/{len(args)}")
        r = time_app(
            n_sub_routers=n_sub_routers,
            n_routes=n_routes,
            n_fields_per_nesting_level=n_fields_per_nesting_level,
            n_nesting_levels=n_nesting_levels,
            use_unique_models=use_unique_models,
            silent=True,
        )
        r |= {
            "n_sub_routers": n_sub_routers,
            "n_routes": n_routes,
            "n_fields_per_nesting_level": n_fields_per_nesting_level,
            "n_nesting_levels": n_nesting_levels,
            "use_unique_models": use_unique_models,
            "n_fields_total": n_fields_per_nesting_level ** (n_nesting_levels + 1) * n_routes,
        }
        print(r)
        all_data.append(r)
    df = pd.DataFrame(all_data)
    print(df)
    fname = f"runtime_data.csv"
    df.to_csv(Path(__file__).parent / fname, index=False)


if __name__ == "__main__":
    main()
