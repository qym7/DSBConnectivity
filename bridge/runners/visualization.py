import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from ..metrics.molecular_metrics import Molecule
from ..utils import PlaceHolder


class Visualizer:
    def __init__(self, dataset_infos):
        self.dataset_infos = dataset_infos
        self.is_molecular = self.dataset_infos.is_molecular

        if self.is_molecular:
            self.remove_h = dataset_infos.remove_h

    def to_networkx_graph(self, graph: PlaceHolder):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        nx_graph = nx.Graph()
        if len(graph.X.shape) == 2:
            graph.X = graph.X.argmax(-1)
        graph.X = graph.X.cpu().numpy()

        for i in range(len(graph.X)):
            nx_graph.add_node(
                i,
                number=i,
                symbol=graph.X[i],
                color_val=graph.X[i],
            )

        adj = graph.E.cpu().numpy()
        if len(adj.shape) == 3:
            adj = adj[:, :, 1:].sum(-1)
        assert len(adj.shape) == 2

        rows, cols = np.where(adj >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            print(edge, adj[0])
            edge_type = adj[edge[0], edge[1]]
            # if edge_type > 100:
            #     import pdb; pdb.set_trace()
            nx_graph.add_edge(
                edge[0],
                edge[1],
                color=float(edge_type),
                weight=3 * edge_type,
            )

        return nx_graph.to_undirected()

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=100):
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        # import pdb; pdb.set_trace()
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        plt.figure()
        nx.draw(
            graph,
            pos,
            font_size=5,
            node_size=node_size,
            with_labels=False,
            node_color=U[:, 1],
            cmap=plt.cm.coolwarm,
            vmin=vmin,
            vmax=vmax,
            edge_color="grey",
        )

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(
        self,
        path: str,
        graphs: PlaceHolder,
        atom_decoder,
        num_graphs_to_visualize: int,
        log="graph",
    ):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        num_graphs = graphs.X.shape[0]
        num_graphs_to_visualize = min(num_graphs_to_visualize, num_graphs)
        if num_graphs_to_visualize > 0:
            print(f"Visualizing {num_graphs_to_visualize} graphs out of {num_graphs}")

        import pdb

        pdb.set_trace()
        graph_list = graphs.split()
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, "graph_{}.png".format(i))

            if self.is_molecular:
                mol = Molecule(graph_list[i], atom_decoder).rdkit_mol
                try:
                    Draw.MolToFile(mol, file_path)
                except rdkit.Chem.KekulizeException:
                    print("Can't kekulize molecule")
            else:
                nx_graph = self.to_networkx_graph(graph_list[i])
                self.visualize_non_molecule(graph=nx_graph, pos=None, path=file_path)

            if wandb.run and log is not None:
                if i < 3:
                    print(f"Saving {file_path} to wandb")
                wandb.log(
                    {log: [wandb.Image(file_path)]},
                    commit=False,
                )

    def visualize_chains(
        self,
        path: str,
        chains: PlaceHolder,
        num_nodes: Tensor,
        local_rank: int,
        num_chains_to_visualize: int,
    ):
        # bs, n_steps, ...
        for i in range(num_chains_to_visualize):  # Iterate over the chains
            # path = os.path.join(chain_path, f"molecule_{batch_id + i}_{local_rank}")
            cur_path = os.path.join(path, f"{i}_graph")
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)

            graphs = []
            chain = PlaceHolder(
                X=chains.X[i, :, : num_nodes[i]].long(),
                E=chains.E[i, :, : num_nodes[i], : num_nodes[i]].long(),
                charge=None,  # chains.charge[:, i, : num_nodes[i]].long(),
                y=None,
            )

            # Iterate over the frames of each molecule
            for j in range(chain.X.shape[0]):
                graph = PlaceHolder(
                    X=chain.X[j],
                    E=chain.E[j],
                    charge=None,
                    y=None,  # chain.charge[j], y=None
                )
                import pdb

                pdb.set_trace()
                if self.is_molecular:
                    graphs.append(
                        Molecule(
                            graph=graph,
                            atom_decoder=self.dataset_infos.atom_decoder,
                        )
                    )
                else:
                    graphs.append(self.to_networkx_graph(graph))

            # Find the coordinates of nodes in the final graph and align all the molecules
            final_graph = graphs[-1]

            if self.is_molecular:
                final_mol = final_graph.rdkit_mol
                AllChem.Compute2DCoords(final_mol)
                coords = []
                for k, atom in enumerate(final_mol.GetAtoms()):
                    positions = final_mol.GetConformer().GetAtomPosition(k)
                    coords.append((positions.x, positions.y, positions.z))

                for graph in graphs:
                    mol = graph.rdkit_mol
                    AllChem.Compute2DCoords(mol)
                    conf = mol.GetConformer()
                    for l, atom in enumerate(mol.GetAtoms()):
                        x, y, z = coords[l]
                        conf.SetAtomPosition(l, Point3D(x, y, z))
            else:
                final_pos = nx.spring_layout(final_graph, seed=0)

            # Visualize and save
            save_paths = []
            image_path = os.path.join(cur_path, "images")
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            for frame in range(len(graphs)):
                file_name = os.path.join(image_path, f"frame_{frame}.png")
                if self.is_molecular:
                    Draw.MolToFile(
                        graphs[frame].rdkit_mol,
                        file_name,
                        size=(300, 300),
                        legend=f"Frame {frame}",
                    )
                else:
                    self.visualize_non_molecule(
                        graph=graphs[frame],
                        pos=final_pos,
                        path=file_name,
                    )
                save_paths.append(file_name)
            print(
                f"{i + 1}/{chains.X.shape[0]} chains saved on local rank {local_rank}.",
                end="",
                flush=True,
            )

            imgs = [imageio.v3.imread(fn) for fn in save_paths]
            cur_folder = path.split("/")[-1]
            gif_path = os.path.join(cur_path, f"{cur_folder}.gif")
            imgs.extend([imgs[-1]] * 10)
            imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)
            if wandb.run:
                wandb.log(
                    {
                        "chain": [
                            wandb.Video(
                                gif_path,
                                caption=gif_path,
                                format="gif",
                            )
                        ]
                    }
                )
                print(f"Saving {gif_path} to wandb")
                wandb.log(
                    {"chain": wandb.Video(gif_path, fps=8, format="gif")},
                    commit=True,
                )
