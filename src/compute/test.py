import torch


def get_paths(A: torch.Tensor, length: int) -> torch.Tensor:
    """Toy version: A^length via repeated @."""
    A = A.float()
    A_paths = A
    for _ in range(length - 1):
        A_paths = A_paths @ A
    return A_paths


class Properties:
    def __init__(self, A: torch.Tensor):
        self.A = A

        # Two properties we care about
        self.paths_count = None
        self.paths_count_flat = None

    def compute_initial(self):
        """Initial computation, e.g. like your compute()."""
        self.paths_count = get_paths(self.A, 6)      # length = 3
        self.paths_count_flat = self.paths_count.flatten()

    def compute_paths_count(self, length: int):
        """Recompute both properties and print them."""
        self.paths_count = get_paths(self.A, length)
        self.paths_count_flat = self.paths_count.flatten()

        # Put a breakpoint on the next line and compare debugger vs prints
        print(f"\n--- After compute_paths_count(length={length}) ---")
        print("paths_count:\n", self.paths_count)
        print("paths_count_flat:", self.paths_count_flat)


if __name__ == "__main__":
    # Simple 3x3 adjacency-like matrix
    A = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])

    p = Properties(A)
    p.compute_initial()              # sets length = 3 internally

    # Now update with a different length
    p.compute_paths_count(5)
    p.compute_paths_count(4)
    p.compute_paths_count(3)
    p.compute_paths_count(1)