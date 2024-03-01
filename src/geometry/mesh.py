"""
mesh.py
"""

from collections import defaultdict
from dataclasses import dataclass, replace

from jaxtyping import Float, Int, jaxtyped
import numpy as np
import torch
from torch import Tensor
from typeguard import typechecked


@dataclass
class Mesh:
    """A mesh"""

    vertices: Float[Tensor, "num_vertex 3"] = None
    """The vertices of the mesh"""
    faces: Int[Tensor, "num_face 3"] = None
    """The faces of the mesh"""
    vertex_colors: Float[Tensor, "num_vertex 3"] = None
    """The vertex colors of the mesh"""
    tex_coordinates: Float[Tensor, "num_vertex 2"] = None
    """The texture coordinates of the mesh"""
    vertex_normals: Float[Tensor, "num_vertex 3"] = None
    """The vertex normals of the mesh"""
    tex_coordinate_indices: Int[Tensor, "num_face 3"] = None
    """The texture indices of the mesh"""
    vertex_normal_indices: Int[Tensor, "num_face 3"] = None
    """The vertex normal indices of the mesh"""
    texture_image: Float[Tensor, "texture_height texture_width 3"] = None
    """The texture image of the mesh"""
    has_texture: bool = False
    """A flag that indicates whether the mesh has a texture"""
    has_normal: bool = False
    """A flag that indicates whether the mesh has vertex normals"""
    device: torch.device = torch.device("cuda")
    """The device where the mesh resides"""

    @jaxtyped(typechecker=typechecked)
    def __post_init__(self) -> None:

        # convert arrays to tensors
        for variable, value in vars(self).items():
            if isinstance(value, np.ndarray):
                setattr(
                    self,
                    variable,
                    torch.from_numpy(value),
                )

        # transfer tensors to device
        for variable, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(
                    self,
                    variable,
                    getattr(self, variable).to(self.device),
                )

        # check whether mesh contains texture
        has_texture = (
            self.texture_image is not None \
            and self.tex_coordinates is not None \
            and self.tex_coordinate_indices is not None
        )
        self.has_texture = has_texture

        # check whether mesh contains vertex normals
        has_normal = (
            self.vertex_normals is not None \
            ####
            # TODO: vertex normal indices are not required during interpolation
            # and self.vertex_normal_indices is not None
            ####
        )
        self.has_normal = has_normal

    def copy(self):
        """
        Returns a copy of the mesh.

        NOTE: Do not forget to copy new attributes as their are appended
        """
        return Mesh(
            vertices=self.vertices.clone().detach(),
            faces=self.faces.clone().detach(),
            vertex_colors=self.vertex_colors.clone().detach() if not self.has_texture else None,
            tex_coordinates=self.tex_coordinates.clone().detach() if self.has_texture else None,
            vertex_normals=self.vertex_normals.clone().detach() if self.has_normal else None,
            tex_coordinate_indices=self.tex_coordinate_indices.clone().detach() if self.has_texture else None,
            vertex_normal_indices=self.vertex_normal_indices.clone().detach() if self.has_normal else None,
            texture_image=self.texture_image.clone().detach() if self.has_texture else None,
            has_texture=self.has_texture,
            has_normal=self.has_normal,
            device=self.device,
        )
    
    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def compute_data_for_arap(self) -> None:
        """
        Computes data needed for ARAP regularization.
        """
        self._compute_one_ring_neighbors()
        self._compute_cotangent_weights()

    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def _compute_one_ring_neighbors(self) -> None:
        """
        Computes 1-ring neighbors of each vertex.

        The code is modified from:
        https://github.com/OllieBoyne/pytorch-arap/blob/master/pytorch_arap/arap.py
        """
        # compute one ring neighbors
        faces = self.faces
        
        orn = defaultdict(set)
        for face in faces:
            for j in [0, 1, 2]:
                i, k = (j + 1) % 3, (j + 2) % 3  # get 2 other vertices
                orn[int(face[j].item())].add(int(face[i].item()))
                orn[int(face[j].item())].add(int(face[k].item()))
        orn = {key: list(value) for key, value in orn.items()}
        self.orn = orn

        # compute (vertex, neighbor vertex, number of neighbor) tuples
        vertices = self.vertices

        ii = []
        jj = []
        nn = []
        for i in range(vertices.shape[0]):
            neighbor_indices = orn[i]
            for n, j in enumerate(neighbor_indices):
                ii.append(i)
                jj.append(j)
                nn.append(n)
    
        self.ii = torch.LongTensor(ii).to(self.device)
        self.jj = torch.LongTensor(jj).to(self.device)
        self.nn = torch.LongTensor(nn).to(self.device)
    
    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def _compute_cotangent_weights(self) -> None:
        """
        Computes cotangent weights of each vertex.

        The code is modified from:
        https://github.com/OllieBoyne/pytorch-arap/blob/master/pytorch_arap/arap.py
        """
        vertices: Float[Tensor, "num_vertex 3"] = self.vertices
        faces: Int[Tensor, "num_face 3"] = self.faces
        face_vertices: Float[Tensor, "num_face 3 3"] = vertices[faces.long()]

        cotangent_weights = torch.zeros(
            (vertices.shape[0], vertices.shape[0])
        ).to(self.device)

        v0, v1, v2 = (
            face_vertices[:, 0],
            face_vertices[:, 1],
            face_vertices[:, 2],
        )

        # Side lengths of each triangle, of shape (sum(F_n),)
		# A is the side opposite v1, B is opposite v2, and C is opposite v3
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)

        # Area of each triangle (with Heron's formula); shape is (F)
        s = 0.5 * (A + B + C)
        # note that the area can be negative (close to 0) causing nans after sqrt()
        # we clip it to a small positive value
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

        # Compute cotangents of angles, of shape (F, 3)
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.stack([cota, cotb, cotc], dim=1).to(self.device)
        cot /= 4.0

        if False:
            if sparse:
                ii = faces[:, [1, 2, 0]]
                jj = faces[:, [2, 0, 1]]
                idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
                w = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))
                w += w.t()
                W.append(w)
            else:
                i = faces[:, [0, 1, 2]].view(-1)  # flattened tensor of by face, v0, v1, v2
                j = faces[:, [1, 2, 0]].view(-1)  # flattened tensor of by face, v1, v2, v0

                # flatten cot, such that the following line sets
                # w_ij = 0.5 * cot a_ij
                W[n][i, j] = 0.5 * cot.view(-1)
                # to include b_ij, simply add the transpose to itself

                ####
                # 20230927 Seungwoo Yoo
                # NOTE: Resolving torch runtime error
                # W[n] += W[n].T
                W[n] += W[n].clone().T
                ####
        else:            
            i = faces[:, [0, 1, 2]].view(-1).long()  # flattened tensor of by face, v0, v1, v2
            j = faces[:, [1, 2, 0]].view(-1).long()  # flattened tensor of by face, v1, v2, v0

            # flatten cot, such that the following line sets
            # w_ij = 0.5 * cot a_ij
            # W[n][i, j] = 0.5 * cot.view(-1)
            cotangent_weights[i, j] = 0.5 * cot.view(-1)
            # to include b_ij, simply add the transpose to itself

            ####
            # 20230927 Seungwoo Yoo
            # NOTE: Resolving torch runtime error
            # W[n] += W[n].T
            # W[n] += W[n].clone().T
            cotangent_weights += cotangent_weights.clone().T
            ####
        self.cotangent_weights = cotangent_weights

        # NOTE: What is this?
        # Compute cotangent matrix in nfmt index format
        orn = self.orn
        max_neighbors = max(map(len, orn.values()))
        num_vertex = vertices.shape[0]

        cotangent_weights_nfmt = torch.zeros(
            (num_vertex, max_neighbors)
        ).to(self.device)

        cotangent_weights_nfmt[self.ii, self.nn] = (
            cotangent_weights[self.ii, self.jj]
        )
        self.cotangent_weights_nfmt = cotangent_weights_nfmt

