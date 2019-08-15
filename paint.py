import os
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import cairo
from scipy.special import softmax
from PIL import Image
from tqdm import tqdm


class Shape:
    """
    A shape params - consists of the location, the height, the width and rgb value
    """
    MAP = {name: i for i, name in enumerate
           (["x", "y", "width", "height", "angle", "red", "green", "blue"])}

    def __init__(
            self,
            vec: np.ndarray,
    ):
        self._vec = vec

    def __getattr__(self, attr: str):
        if attr in Shape.MAP:
            return float(self.__getattribute__("_vec")[Shape.MAP[attr]])
        else:
            return self.__getattribute__(attr)

    def as_vec(self) -> np.ndarray:
        return self._vec

    @classmethod
    def from_params(
            cls,
            x: float,
            y: float,
            width: float,
            height: float,
            angle: float,
            red: float,
            green: float,
            blue: float,
        ) -> 'Shape':
        return cls(np.asarray([x, y, width, height, angle, red, green, blue]))


class Painting:
    """
    A painting is a list of shapes
    """
    def __init__(self, width: int, height: int, shape_params: np.ndarray):
        self.width = width
        self.height = height
        self.params = shape_params
        self.shapes = [Shape(shape_params[i]) for i in range(len(shape_params))]

    @classmethod
    def random(cls, width: int, height: int, num_shapes: int) -> 'Painting':
        vec= np.random.random((num_shapes, 8))
        return cls(width, height, vec)

    def render(self) -> np.ndarray:
        # sort the shapes
        shapes = sorted(self.shapes, key=lambda x: x.height * x.width, reverse=True)

        # create a new surface
        surface = cairo.ImageSurface(cairo.Format.RGB24, self.width, self.height)

        # add a background
        ctx = cairo.Context(surface)
        ctx.scale(self.width, self.height)
        ctx.rectangle(0, 0, 1, 1)
        ctx.set_source_rgb(0.0, 0.0, 0.0)
        ctx.fill()

        # add each shape
        for s in shapes:
            ctx.identity_matrix()
            ctx.translate(s.x * self.width, s.y * self.height)
            ctx.rotate(s.angle * np.pi * 2.0)
            try:
                ctx.scale(self.width * s.width, self.height * s.height)
            except:
                import pdb; pdb.set_trace()
            ctx.rectangle(0, 0, 1, 1)
            ctx.set_source_rgb(s.red, s.green, s.blue)
            ctx.fill()

        buf = surface.get_data()
        data = np.ndarray(
            shape=(self.height, self.width, 4),
            dtype=np.uint8,
            buffer=buf
        )
        data = data.astype(np.float32) / 255.0
        return data, surface
    
    def random_neighbors(self, n: int, stddev: float) -> Tuple[List['Painting'], np.ndarray]:
        """
        Generate some random neighbors for this painting
        """
        angle_dim = Shape.MAP["angle"]
        # generate a sample with norm
        sample = np.random.normal(size=[n, *self.params.shape])
        while True:
            new_params = np.expand_dims(self.params, axis=0) + (sample * stddev)
            should_resample = np.logical_or(new_params > 1.0, new_params < 0.0)
            # never resample angles
            should_resample[:, :, angle_dim] = False
            resample_indices = np.argwhere(should_resample)
            if len(resample_indices) > 0:
                ridx = np.transpose(resample_indices)
                sample[ridx[0], ridx[1], ridx[2]] = np.random.normal(size=(len(resample_indices)))
            else:
                break

        # Now sort the shapes
        #sizes = new_params[:, :, Shape.MAP["width"]] * new_params[:, :, Shape.MAP["height"]]
        #sort_idx = np.argsort(sizes, axis=-1)[:,::-1]
        #outer_dim_idx = [[i] for i in range(sizes.shape[0])]
        #inner_dim_idx = [sort_idx[i] for i in range(sizes.shape[0])]
        #sorted_params = new_params[outer_dim_idx, inner_dim_idx]

        # mod the angle
        new_params[:, :, angle_dim] %= (2.0 * np.pi)
        sample[:, :, angle_dim] = (new_params[:, :, angle_dim] - np.expand_dims(self.params[:,angle_dim], axis=0)) / stddev

        new_paintings = [Painting(self.width, self.height, params) for params in new_params]
        return new_paintings, sample * stddev
    

class PaintingPopulation:
    """
    A population of paintings
    """
    def __init__(self, width: int, height: int, paintings: List[Painting], num_shapes: int):
        self.width = width
        self.height = height
        self.paintings = paintings
        self.num_shapes = num_shapes

    @classmethod
    def random(cls, width: int, height: int, n: int, num_shapes: int) -> 'PaintingPopulation':
        # generate a bunch of random paintings
        paintings = [Painting.random(width, height, num_shapes) for _ in range(n)]
        return cls(width, height, paintings, num_shapes)

class Trainer:
    def __init__(
        self,
        original: np.ndarray,
        painting: Painting,
        population_size: int=100,
        alpha: float=1.0,
        stddev: float=0.1,
        save_dir: str = None
    ):
        """
        original: The original image as a [H x W x 4] BGRA image
        painting: The current best painting
        """
        self.original = original
        self.painting = painting
        if self.painting.width != self.original.shape[1] or self.painting.height != self.original.shape[0]:
            raise ValueError("Size mismatch!")
        self.population_size = population_size
        self.alpha = alpha
        self.stddev = stddev
        self.save_dir = save_dir
    
    def checkpoint_image_path(self, name: str) -> Optional[str]:
        if self.save_dir is not None:
            path = os.path.join(self.save_dir, "checkpoint_img")
            os.makedirs(path, exist_ok=True)
            return os.path.join(path, name)
        else:
            return None
    
    def train(self, steps: int, checkpoint_freq: int = 10, save_dir: str = None):
        for i in tqdm(range(steps)):
            self.step()
            # Now checkpoint
            if i % checkpoint_freq == 0 and self.save_dir is not None:
                dat, surface = self.painting.render()
                loss = self.loss(dat)

                surface.write_to_png(self.checkpoint_image_path(f"step_{i}_loss_{loss:.3f}.png"))

    def step(self):
        # generate a new population by getting neighbors for best painting
        population, diff = self.painting.random_neighbors(self.population_size, self.stddev)

        # weighted sum to find the best
        curr_loss = self.loss(self.painting.render()[0])
        rewards = curr_loss - np.asarray([self.loss(p.render()[0]) for p in population])
        weights = rewards / float(len(rewards))
        best_params = self.painting.params + (np.sum(np.reshape(weights, (-1, 1, 1)) * diff, axis=0) * self.alpha)

        # clip the params
        angles = best_params[:, Shape.MAP["angle"]] % (2.0 * np.pi)
        best_params = np.clip(best_params, 0.01, 1.0)
        best_params[:, Shape.MAP["angle"]] = angles

        # create a new painting with the best params
        self.painting = Painting(self.painting.width, self.painting.height, best_params)
        # self.painting = population[np.argmax(rewards)]

    def loss(self, source: np.ndarray) -> float:
        """
        L1 Loss
        """
        return np.sum(np.abs(self.original - source)) / (self.original.shape[0] * self.original.shape[1] * 4)

def load_image(path: str) -> np.ndarray:
    im = Image.open(path)
    arr = np.asarray(im)
    arr = arr[:, :, ::-1] # BGR (to be the same as cairo)
    arr = np.concatenate(
            [
                arr,
                np.tile(np.asarray([255], dtype=np.uint8), (arr.shape[0], arr.shape[1], 1))
            ],
            axis=-1
    )
    arr = arr.astype(np.float32) / 255.0
    return arr

def main(args):
    original = load_image(args.image_target)
    trainer = Trainer(
        original=original,
        painting=Painting.random(width=original.shape[1], height=original.shape[0], num_shapes=args.num_shapes),
        population_size=args.population_size,
        alpha=args.alpha,
        stddev=args.stddev,
        save_dir=args.save_dir,
    )
    trainer.train(args.iterations)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Guess and Paint")
    parser.add_argument("-i", "--image-target", type=str, required=True, help="The target image")
    parser.add_argument("-n", "--num-shapes", type=int, required=True, help="The number of shapes")
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="The learning rate")
    parser.add_argument("-s", "--stddev", type=float, default=0.1, help="standard deviation")
    parser.add_argument("-p", "--population-size", type=int, default=100, help="The number of random neighbors to explore at each iteration")
    parser.add_argument("--iterations", type=int, default=1000, help="The number of iterations to train for")
    parser.add_argument("--save-dir", type=str, default=None, help="The directory to save outputs to")
    parser.add_argument("--config", type=str, help="If provided, it will override any existing options with this config")

    ARGS = parser.parse_args()
    import json
    print(json.dumps(vars(ARGS), indent=4, sort_keys=True))

    if ARGS.config is not None:
        raise NotImplementedError
    if ARGS.save_dir is not None and os.path.exists(ARGS.save_dir):
        print("Save Dir already exists!")
        command = input("y to override> ")
        if command != "y":
            exit(1)
        else:
            import shutil
            shutil.rmtree(ARGS.save_dir)
    main(ARGS)
