from __future__ import annotations
import random
from collections import deque
from .model import CARDINAL_DIRECTIONS, Direction, Maze, MazeError

WallCandidate = tuple[int, int, Direction]
Neighbor = tuple[Direction, tuple[int, int]]
SUPPORTED_ALGORITHMS: tuple[str, ...] = ("dfs", "hunt_and_kill")
Parent = tuple[tuple[int, int], str] | None
Position = tuple[int, int]
Edge = frozenset  # frozenset({celda_a, celda_b}): arista sin dirección
LETTERS: dict[Direction, str] = {
    Direction.NORTH: "N",
    Direction.EAST: "E",
    Direction.SOUTH: "S",
    Direction.WEST: "W",
}
DELTAS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}


class BaseAlgorithm:
    def __init__(
        self,
        maze: Maze,
        rng: random.Random,
        start: tuple[int, int] | None = None,
        finish: tuple[int, int] | None = None,
    ) -> None:
        self.maze = maze
        self.rng = rng
        self.start = start
        self.visited: set[tuple[int, int]] = set()
        self.finish = finish

    def generate(self) -> None:
        """Método principal que lanza el algoritmo (Fachada)."""
        self._carve()

    def _carve(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Comprobación de bloques 3x3 abiertos
    # ------------------------------------------------------------------
    def _search_square(self, x: int, y: int) -> bool:
        for c in range(y, y + 2):
            for l in range(x, x + 2):
                cell = self.maze.get_cell(l, c)
                if cell.has_wall(Direction.EAST) or cell.has_wall(Direction.SOUTH):
                    return True
        return False

    def _makes_3_square(self, x: int, y: int, direction: Direction) -> bool:
        neighbor = self.maze.get_neighbor(x, y, direction)
        if neighbor is None:
            return False
        nx, ny = neighbor

        if not self.maze.open_wall(x, y, direction):
            return False

        min_x = max(0, min(x, nx) - 2)
        min_y = max(0, min(y, ny) - 2)
        max_x = min(self.maze.width - 3, max(x, nx))
        max_y = min(self.maze.height - 3, max(y, ny))

        creates = False
        for sy in range(min_y, max_y + 1):
            for sx in range(min_x, max_x + 1):
                if not self._search_square(sx, sy):
                    creates = True
                    break
            if creates:
                break

        self.maze.close_wall(x, y, direction)
        return creates

    # ------------------------------------------------------------------
    # Braid: aleatorio + garantía de dos caminos entre entrada y salida
    # ------------------------------------------------------------------
    def braid(self, p_break: float = 0.45) -> bool:
        """Rompe muros al azar (p_break) y después garantiza que entre la
        entrada y la salida haya al menos dos caminos independientes.

        Returns:
            True si hay dos caminos independientes, False si no ha sido
            posible (laberinto demasiado pequeño / restricción 3x3).
        """
        for cy in range(self.maze.height):
            for cx in range(self.maze.width):
                if (cx, cy) in self.maze.pattern42_cells:
                    continue

                if self.rng.random() < p_break:
                    directions = list(CARDINAL_DIRECTIONS)
                    direction = self.rng.choice(directions)
                    neighbor = self.maze.get_neighbor(cx, cy, direction)

                    if neighbor is None or neighbor in self.maze.pattern42_cells:
                        continue

                    if self.maze.has_wall(cx, cy, direction) and not self._makes_3_square(cx, cy, direction):
                        self.maze.open_wall(cx, cy, direction)
        return self._ensure_two_paths()

    def _endpoints(self) -> tuple[Position, Position]:
        origin = self.maze.entry if self.start is None else self.start
        destination = self.maze.exit if self.finish is None else self.finish
        return origin, destination


    def solve(self, blocked: set[Edge] | None = None) -> str:
        origin = self.maze.entry if self.start is None else self.start
        destination = self.maze.exit if self.finish is None else self.finish

        parents: dict[Position, Parent] = {origin: None}
        queue: deque[Position] = deque([origin])
        while queue:
            current = queue.popleft()
            if current == destination:
                break
            for direction in CARDINAL_DIRECTIONS:
                neighbor = self.maze.get_neighbor(
                    current[0],
                    current[1],
                    direction
                )
                if neighbor is None: continue
                if neighbor in self.maze.pattern42_cells: continue
                if neighbor in parents: continue
                if self.maze.has_wall(current[0], current[1], direction): continue
                if blocked and frozenset((current, neighbor)) in blocked: continue
                parents[neighbor] = (current, LETTERS[direction])
                queue.append(neighbor)

        if destination not in parents:
            raise MazeError(
                "There is no existing path between the start and the exit"
            )

        steps: list[str] = []
        node = destination
        while node != origin:
            parent = parents[node]
            if parent is None:
                raise MazeError(
                    "It was impossible to rebuild the shortests path"
                )
            previous, letter = parent
            steps.append(letter)
            node = previous
        steps.reverse()
        return "".join(steps)

    def _bridges_on_path(self) -> list[Edge] | None:
        try:
            steps = self.solve()
        except MazeError:
            return None          # E y X no están conectados

        current, _ = self._endpoints()
        edges: list[Edge] = []
        for letter in steps:
            dx, dy = DELTAS[letter]
            nxt = (current[0] + dx, current[1] + dy)
            edges.append(frozenset((current, nxt)))
            current = nxt

        bridges: list[Edge] = []
        for edge in edges:
            try:
                self.solve({edge})   # si llega, la arista no es puente
            except MazeError:
                bridges.append(edge)
        return bridges

    def _ensure_two_paths(self) -> bool:
        bridges = self._bridges_on_path()
        if bridges is None:
            return False
        if not bridges:
            return True

        candidates: list[WallCandidate] = []
        for cy in range(self.maze.height):
            for cx in range(self.maze.width):
                if (cx, cy) in self.maze.pattern42_cells:
                    continue
                for direction in (Direction.EAST, Direction.SOUTH):
                    neighbor = self.maze.get_neighbor(cx, cy, direction)
                    if (
                        neighbor is not None
                        and neighbor not in self.maze.pattern42_cells
                        and self.maze.has_wall(cx, cy, direction)
                    ):
                        candidates.append((cx, cy, direction))
        self.rng.shuffle(candidates)

        for cx, cy, direction in candidates:
            if self._makes_3_square(cx, cy, direction):
                continue                      # otro muro lo hizo inseguro
            self.maze.open_wall(cx, cy, direction)
            new_bridges = self._bridges_on_path()
            if new_bridges is not None and len(new_bridges) < len(bridges):
                bridges = new_bridges
                if not bridges:
                    return True
            else:
                self.maze.close_wall(cx, cy, direction)
        return not bridges

    def _resolve_start(self) -> tuple[int, int]:
        candidate = self.maze.entry if self.start is None else self.start
        if (
            self.maze.is_inside(*candidate)
            and candidate not in self.maze.pattern42_cells
        ):
            return candidate
        cells = self.maze.non_pattern_cells()
        if not cells:
            raise MazeError("No hay celdas libres para generar el laberinto.")
        return cells[0]


class DFS(BaseAlgorithm):
    def _carve(self) -> None:
        """Generador DFS del usuario."""
        current = self._resolve_start()
        self.visited = {current}
        stack: list[tuple[int, int]] = [current]

        while stack:
            x, y = stack[-1]
            directions = list(CARDINAL_DIRECTIONS)
            self.rng.shuffle(directions)

            moved = False
            for direction in directions:
                neighbor = self.maze.get_neighbor(x, y, direction)

                if neighbor is None or neighbor in self.maze.pattern42_cells:
                    continue

                if neighbor in self.visited:
                    continue

                self.maze.open_wall(x, y, direction)
                self.visited.add(neighbor)
                stack.append(neighbor)
                moved = True
                break

            if not moved:
                stack.pop()


class HuntAndKill(BaseAlgorithm):
    def _unvisited_neighbors(self, position: tuple[int, int]) -> list[Neighbor]:
        x, y = position
        result: list[Neighbor] = []
        for direction in CARDINAL_DIRECTIONS:
            neighbor = self.maze.get_neighbor(x, y, direction)
            if neighbor is None:
                continue
            if neighbor in self.maze.pattern42_cells:
                continue
            if neighbor in self.visited:
                continue
            result.append((direction, neighbor))
        return result

    def _visited_neighbors(self, position: tuple[int, int]) -> list[Neighbor]:
        x, y = position
        result: list[Neighbor] = []
        for direction in CARDINAL_DIRECTIONS:
            neighbor = self.maze.get_neighbor(x, y, direction)
            if neighbor is None:
                continue
            if neighbor in self.maze.pattern42_cells:
                continue
            if neighbor in self.visited:
                result.append((direction, neighbor))
        return result

    def _hunt(self) -> tuple[int, int] | None:
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                position = (x, y)
                if position in self.visited:
                    continue
                if position in self.maze.pattern42_cells:
                    continue
                candidates = self._visited_neighbors(position)
                if not candidates:
                    continue
                direction, _ = self.rng.choice(candidates)
                self.maze.open_wall(x, y, direction)
                self.visited.add(position)
                return position
        return None

    def _carve(self) -> None:
        """Carve a perfect maze with the hunt-and-kill algorithm."""
        current = self._resolve_start()
        self.visited = {current}
        while True:
            candidates = self._unvisited_neighbors(current)
            if candidates:
                direction, neighbor = self.rng.choice(candidates)
                self.maze.open_wall(current[0], current[1], direction)
                self.visited.add(neighbor)
                current = neighbor
                continue
            hunted = self._hunt()
            if hunted is None:
                return
            current = hunted


class BFS(BaseAlgorithm):

    def _carve(self) -> None:
        ...


class AlgorithmFactory:
    @classmethod
    def get_algorithm(
        cls,
        maze: Maze,
        algorithm: str,
        rng: random.Random,
        start: tuple[int, int] | None,
        finish: tuple[int, int] | None = None,
    ) -> BaseAlgorithm:
        if algorithm == "dfs":
            return DFS(maze, rng, start, finish)
        if algorithm == "hunt_and_kill":
            return HuntAndKill(maze, rng, start, finish)
        raise MazeError(
            f"Algoritmo desconocido: {algorithm!r}. "
            f"Use uno de: {', '.join(SUPPORTED_ALGORITHMS)}."
        )