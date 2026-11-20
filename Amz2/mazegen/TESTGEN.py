import unittest
from collections import deque

from .generator import MazeGenerator
from .model import CARDINAL_DIRECTIONS, Direction, Maze, MazeError

LETTERS_TO_DIR: dict[str, Direction] = {
    "N": Direction.NORTH,
    "E": Direction.EAST,
    "S": Direction.SOUTH,
    "W": Direction.WEST,
}

ALGORITHMS = ("dfs", "hunt_and_kill")
SEEDS = (1, 7, 42, 99, 2026)

# Fragmento del aviso que añade generate() cuando braid() devuelve False.
# last_warning junta varios avisos con " | ", así que se busca con "in".
BRAID_WARNING = "dos rutas independientes"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def verify_path(maze: Maze, path_str: str, start, finish) -> bool:
    """Comprueba paso a paso que la solución es válida y llega a la salida."""
    x, y = start
    for char in path_str:
        if char not in LETTERS_TO_DIR:
            return False
        direction = LETTERS_TO_DIR[char]
        if maze.has_wall(x, y, direction):
            return False
        neighbor = maze.get_neighbor(x, y, direction)
        if neighbor is None or neighbor in maze.pattern42_cells:
            return False
        x, y = neighbor
    return (x, y) == finish


def find_path(maze: Maze, blocked=None):
    """BFS independiente del código bajo test. Devuelve lista de celdas o None."""
    start, goal = maze.entry, maze.exit
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for d in CARDINAL_DIRECTIONS:
            nxt = maze.get_neighbor(cur[0], cur[1], d)
            if nxt is None or nxt in prev or nxt in maze.pattern42_cells:
                continue
            if maze.has_wall(cur[0], cur[1], d):
                continue
            if blocked and frozenset((cur, nxt)) in blocked:
                continue
            prev[nxt] = cur
            queue.append(nxt)
    if goal not in prev:
        return None
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


def has_two_paths(maze: Maze) -> bool:
    """True si ninguna arista del camino E->X es imprescindible."""
    path = find_path(maze)
    if path is None:
        return False
    for edge in (frozenset(p) for p in zip(path, path[1:])):
        if find_path(maze, {edge}) is None:
            return False
    return True


def wall_signature(maze: Maze) -> tuple:
    """Huella de todos los muros del laberinto (para comparar semillas)."""
    return tuple(
        (maze.has_wall(x, y, Direction.EAST), maze.has_wall(x, y, Direction.SOUTH))
        for y in range(maze.height)
        for x in range(maze.width)
    )


# ----------------------------------------------------------------------
# Validación de argumentos
# ----------------------------------------------------------------------
class TestGeneratorValidation(unittest.TestCase):

    def make(self, **overrides):
        args = dict(width=11, height=9, entry=(0, 0), exit=(10, 8))
        args.update(overrides)
        return MazeGenerator(**args)

    def test_valid_arguments_do_not_raise(self) -> None:
        self.make()

    def test_invalid_width(self) -> None:
        for bad in (0, -3, True, 11.0, "11", None):
            with self.subTest(width=bad):
                with self.assertRaises(MazeError):
                    self.make(width=bad)

    def test_invalid_height(self) -> None:
        for bad in (0, -3, True, 9.0, "9", None):
            with self.subTest(height=bad):
                with self.assertRaises(MazeError):
                    self.make(height=bad)

    def test_invalid_seed(self) -> None:
        for bad in (True, 1.5, "abc"):
            with self.subTest(seed=bad):
                with self.assertRaises(MazeError):
                    self.make(seed=bad)

    def test_unknown_algorithm(self) -> None:
        with self.assertRaises(MazeError):
            self.make(algorithm="prim")

    def test_entry_and_exit_out_of_bounds(self) -> None:
        with self.assertRaises(MazeError):
            self.make(entry=(11, 0))
        with self.assertRaises(MazeError):
            self.make(entry=(0, 9))
        with self.assertRaises(MazeError):
            self.make(entry=(-1, 0))
        with self.assertRaises(MazeError):
            self.make(exit=(10, 9))

    def test_entry_equal_to_exit(self) -> None:
        with self.assertRaises(MazeError):
            self.make(entry=(3, 3), exit=(3, 3))

    def test_malformed_positions(self) -> None:
        for bad in ((0,), (0, 0, 0), ("a", 0), (0, 1.5), (True, 0)):
            with self.subTest(entry=bad):
                with self.assertRaises(MazeError):
                    self.make(entry=bad)
            with self.subTest(exit=bad):
                with self.assertRaises(MazeError):
                    self.make(exit=bad)


# ----------------------------------------------------------------------
# Semilla
# ----------------------------------------------------------------------
class TestGeneratorSeed(unittest.TestCase):

    def test_seed_property_returns_given_seed(self) -> None:
        gen = MazeGenerator(11, 9, (0, 0), (10, 8), seed=5)
        self.assertEqual(gen.seed, 5)

    def test_seed_is_generated_when_none(self) -> None:
        gen = MazeGenerator(11, 9, (0, 0), (10, 8), seed=None)
        self.assertIsInstance(gen.seed, int)
        self.assertGreaterEqual(gen.seed, 0)
        self.assertLess(gen.seed, 2**32)

    def test_same_seed_gives_same_maze(self) -> None:
        for algo in ALGORITHMS:
            for perfect in (True, False):
                with self.subTest(algorithm=algo, perfect=perfect):
                    m1 = MazeGenerator(11, 9, (0, 0), (10, 8), perfect=perfect,
                                       seed=123, algorithm=algo).generate()
                    m2 = MazeGenerator(11, 9, (0, 0), (10, 8), perfect=perfect,
                                       seed=123, algorithm=algo).generate()
                    self.assertEqual(wall_signature(m1), wall_signature(m2))

    def test_different_seeds_give_different_mazes(self) -> None:
        signatures = {
            wall_signature(
                MazeGenerator(21, 19, (0, 0), (20, 18), seed=s).generate()
            )
            for s in SEEDS
        }
        self.assertGreater(len(signatures), 1)


# ----------------------------------------------------------------------
# Estado antes de generar
# ----------------------------------------------------------------------
class TestGeneratorBeforeGenerate(unittest.TestCase):

    def setUp(self) -> None:
        self.gen = MazeGenerator(11, 9, (0, 0), (10, 8), seed=1)

    def test_get_maze_before_generate_raises(self) -> None:
        with self.assertRaises(MazeError):
            self.gen.get_maze()

    def test_solve_before_generate_raises(self) -> None:
        with self.assertRaises(MazeError):
            self.gen.solve()

    def test_initial_state(self) -> None:
        self.assertFalse(self.gen.pattern42_present)
        self.assertIsNone(self.gen.last_warning)


# ----------------------------------------------------------------------
# Laberinto perfecto
# ----------------------------------------------------------------------
class TestGeneratorPerfect(unittest.TestCase):

    def test_perfect_maze_properties(self) -> None:
        for algo in ALGORITHMS:
            for seed in SEEDS:
                with self.subTest(algorithm=algo, seed=seed):
                    gen = MazeGenerator(11, 9, (0, 0), (10, 8), perfect=True,
                                        seed=seed, algorithm=algo)
                    maze = gen.generate()

                    self.assertIs(maze, gen.get_maze())
                    self.assertEqual(maze.count_loops(), 0)
                    self.assertTrue(maze.is_fully_connected())
                    self.assertFalse(maze.has_open_3x3())

    def test_perfect_maze_has_valid_solution(self) -> None:
        for algo in ALGORITHMS:
            with self.subTest(algorithm=algo):
                gen = MazeGenerator(21, 19, (0, 0), (20, 18), perfect=True,
                                    seed=42, algorithm=algo)
                maze = gen.generate()
                solution = gen.solve()
                self.assertGreater(len(solution), 0)
                self.assertTrue(verify_path(maze, solution, (0, 0), (20, 18)))

    def test_perfect_maze_has_unique_path(self) -> None:
        """En un perfecto TODAS las aristas del camino son puentes."""
        gen = MazeGenerator(11, 9, (0, 0), (10, 8), perfect=True, seed=42)
        maze = gen.generate()
        self.assertFalse(has_two_paths(maze))


# ----------------------------------------------------------------------
# Laberinto jugable (braid)
# ----------------------------------------------------------------------
class TestGeneratorBraid(unittest.TestCase):

    def test_braid_maze_properties(self) -> None:
        for algo in ALGORITHMS:
            for seed in SEEDS:
                with self.subTest(algorithm=algo, seed=seed):
                    gen = MazeGenerator(21, 19, (0, 0), (20, 18), perfect=False,
                                        seed=seed, algorithm=algo)
                    maze = gen.generate()

                    self.assertGreater(maze.count_loops(), 0)
                    self.assertTrue(maze.is_fully_connected())
                    self.assertFalse(maze.has_open_3x3())

    def test_braid_maze_has_valid_solution(self) -> None:
        for algo in ALGORITHMS:
            for seed in SEEDS:
                with self.subTest(algorithm=algo, seed=seed):
                    gen = MazeGenerator(11, 9, (0, 0), (10, 8), perfect=False,
                                        seed=seed, algorithm=algo)
                    maze = gen.generate()
                    solution = gen.solve()
                    self.assertTrue(
                        verify_path(maze, solution, (0, 0), (10, 8))
                    )

    def test_braid_guarantees_two_paths_or_warns(self) -> None:
        """Si hay dos caminos no debe haber aviso de braid; si no, sí."""
        for algo in ALGORITHMS:
            for seed in SEEDS:
                with self.subTest(algorithm=algo, seed=seed):
                    gen = MazeGenerator(11, 9, (0, 0), (10, 8), perfect=False,
                                        seed=seed, algorithm=algo)
                    maze = gen.generate()
                    warning = gen.last_warning or ""
                    if has_two_paths(maze):
                        self.assertNotIn(BRAID_WARNING, warning)
                    else:
                        self.assertIn(BRAID_WARNING, warning)

    def test_braid_two_paths_on_large_maze(self) -> None:
        """En un laberinto grande SIEMPRE tiene que haber dos caminos."""
        for algo in ALGORITHMS:
            for seed in SEEDS:
                with self.subTest(algorithm=algo, seed=seed):
                    gen = MazeGenerator(21, 19, (0, 0), (20, 18), perfect=False,
                                        seed=seed, algorithm=algo)
                    maze = gen.generate()
                    self.assertTrue(has_two_paths(maze))
                    self.assertNotIn(BRAID_WARNING, gen.last_warning or "")

    def test_small_maze_with_logo_two_paths_9x7(self) -> None:
        """El caso difícil: 9x7 con el logo en medio (1 fila arriba y 1 abajo)."""
        for algo in ALGORITHMS:
            for seed in range(30):
                with self.subTest(algorithm=algo, seed=seed):
                    gen = MazeGenerator(9, 7, (0, 0), (8, 6), perfect=False,
                                        seed=seed, algorithm=algo)
                    maze = gen.generate()
                    warning = gen.last_warning or ""

                    self.assertTrue(gen.pattern42_present)
                    self.assertTrue(maze.is_fully_connected())
                    self.assertFalse(maze.has_open_3x3())
                    if has_two_paths(maze):
                        self.assertNotIn(BRAID_WARNING, warning)
                    else:
                        self.assertIn(BRAID_WARNING, warning)

    def test_small_maze_with_logo_always_two_paths_9x7(self) -> None:
        """Versión estricta: en 9x7 SIEMPRE debería haber dos caminos."""
        for algo in ALGORITHMS:
            for seed in range(30):
                with self.subTest(algorithm=algo, seed=seed):
                    gen = MazeGenerator(9, 7, (0, 0), (8, 6), perfect=False,
                                        seed=seed, algorithm=algo)
                    maze = gen.generate()
                    self.assertTrue(has_two_paths(maze))


# ----------------------------------------------------------------------
# Patrón 42
# ----------------------------------------------------------------------
class TestGeneratorPattern42(unittest.TestCase):

    def test_logo_is_present_and_closed(self) -> None:
        gen = MazeGenerator(11, 9, (0, 0), (10, 8), seed=42)
        maze = gen.generate()

        self.assertTrue(gen.pattern42_present)
        self.assertGreater(len(maze.pattern42_cells), 0)
        for x, y in maze.pattern42_cells:
            self.assertTrue(maze.get_cell(x, y).is_fully_closed())

    def test_entry_and_exit_not_in_logo(self) -> None:
        maze = MazeGenerator(11, 9, (0, 0), (10, 8), seed=42).generate()
        self.assertNotIn((0, 0), maze.pattern42_cells)
        self.assertNotIn((10, 8), maze.pattern42_cells)

    def test_logo_disabled(self) -> None:
        gen = MazeGenerator(11, 9, (0, 0), (10, 8), seed=42, pattern42=False)
        maze = gen.generate()

        self.assertFalse(gen.pattern42_present)
        self.assertEqual(len(maze.pattern42_cells), 0)
        self.assertTrue(maze.is_fully_connected())

    def test_logo_does_not_fit_in_tiny_maze(self) -> None:
        """En un 3x3 no cabe el logo: se omite y hay aviso."""
        gen = MazeGenerator(3, 3, (0, 0), (2, 2), perfect=True, seed=1)
        maze = gen.generate()

        self.assertFalse(gen.pattern42_present)
        self.assertEqual(len(maze.pattern42_cells), 0)
        self.assertIsNotNone(gen.last_warning)

    def test_logo_needs_at_least_9x7(self) -> None:
        """El mínimo es 9 de ancho x 7 de alto; 7x9 NO cuenta."""
        fits = MazeGenerator(9, 7, (0, 0), (8, 6), perfect=True, seed=1)
        fits.generate()
        self.assertTrue(fits.pattern42_present)

        swapped = MazeGenerator(7, 9, (0, 0), (6, 8), perfect=True, seed=1)
        swapped.generate()
        self.assertFalse(swapped.pattern42_present)

    def test_solution_never_crosses_logo(self) -> None:
        for algo in ALGORITHMS:
            with self.subTest(algorithm=algo):
                gen = MazeGenerator(11, 9, (0, 0), (10, 8), seed=7,
                                    algorithm=algo)
                maze = gen.generate()
                x, y = (0, 0)
                for char in gen.solve():
                    x, y = maze.get_neighbor(x, y, LETTERS_TO_DIR[char])
                    self.assertNotIn((x, y), maze.pattern42_cells)


# ----------------------------------------------------------------------
# Entrada y salida personalizadas
# ----------------------------------------------------------------------
class TestGeneratorCustomEndpoints(unittest.TestCase):

    def test_solution_respects_custom_entry_and_exit(self) -> None:
        gen = MazeGenerator(21, 19, (0, 18), (20, 0), perfect=True, seed=3)
        maze = gen.generate()
        solution = gen.solve()
        self.assertTrue(verify_path(maze, solution, (0, 18), (20, 0)))


if __name__ == "__main__":
    unittest.main()