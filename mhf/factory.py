"""
Factory methods for creating MHF-optimized models
"""

from typing import Any, Dict, Optional, Type, Union

import torch
import torch.nn as nn

from neuralop.models import (
    FNO,
    TFNO,
    SFNO,
    UNO,
    UQNO,
    FNOGNO,
    GINO,
    CODANO,
    RNO,
    LocalNO,
    OTNO,
    get_model,
)

from .spectral_mhf import SpectralConvMHF
from ..models.fno_mhf import MHFNO, MHFTFNO
from ..models.uno_mhf import MHFUNO
from ..models.gino_mhf import MHF_GINO, MHFFNOGNO
from ..models.codano_mhf import MHFCODANO
from ..models.rno_mhf import MHFRNO
from ..models.localno_mhf import MHFLocalNO
from ..models.otno_mhf import MHFOTNO
from ..models.sfno_mhf import MHSFNO


_MODEL_REGISTRY = {
    # FNO family
    "fno": MHFNO,
    "mhfno": MHFNO,
    "mhf-fno": MHFNO,
    
    "tfno": MHFTFNO,
    "mhftfno": MHFTFNO,
    "mhf-tfno": MHFTFNO,
    
    "sfno": MHSFNO,
    "mhfsfno": MHSFNO,
    "mhf-sfno": MHSFNO,
    
    "uno": MHFUNO,
    "mhfuno": MHFUNO,
    "mhf-uno": MHFUNO,
    
    # Other operators
    "gino": MHF_GINO,
    "mhfgino": MHF_GINO,
    "mhf-gino": MHF_GINO,
    
    "fnogno": MHFFNOGNO,
    "mhffnogno": MHFFNOGNO,
    "mhf-fnogno": MHFFNOGNO,
    
    "codano": MHFCODANO,
    "mhfcodano": MHFCODANO,
    "mhf-codano": MHFCODANO,
    
    "rno": MHFRNO,
    "mhfrno": MHFRNO,
    "mhf-rno": MHFRNO,
    
    "localno": MHFLocalNO,
    "mhflocalno": MHFLocalNO,
    "mhf-localno": MHFLocalNO,
    
    "otno": MHFOTNO,
    "mhfotno": MHFOTNO,
    "mhf-otno": MHFOTNO,
}


def register_mhf_model(name: str, model_class: Type[nn.Module]) -> None:
    """Register a new MHF model with the factory
    
    Parameters
    ----------
    name : str
        Name to register the model under
    model_class : Type[nn.Module]
        Model class to register
    """
    _MODEL_REGISTRY[name.lower()] = model_class


def get_mhf_model(
    model_name: str,
    **kwargs,
) -> nn.Module:
    """Get an MHF-optimized model by name
    
    Parameters
    ----------
    model_name : str
        Name of the MHF-optimized model
    **kwargs
        Additional model configuration parameters
        
    Returns
    -------
    nn.Module
        MHF-optimized model
        
    Examples
    --------
    >>> model = get_mhf_model("fno", n_modes=(16, 16), in_channels=1, 
    ...                       out_channels=1, hidden_channels=64, 
    ...                       mhf_rank=8)
    """
    model_name = model_name.lower().replace("-", "").replace("_", "")
    
    if model_name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_name](**kwargs)
    else:
        raise ValueError(
            f"Unknown MHF model: {model_name}. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )


def compress_pretrained_model(
    model: nn.Module,
    mhf_rank: Union[int, Dict[str, Union[int, list]]],
    factorization: str = "tucker",
    implementation: str = "factorized",
) -> nn.Module:
    """Compress a pre-trained original neuraloperator model with MHF
    
    This method takes an already-trained original model from neuraloperator
    and applies MHF compression to all applicable layers.
    
    Parameters
    ----------
    model : nn.Module
        Original pre-trained model from neuraloperator
    mhf_rank : Union[int, Dict[str, Union[int, list]]]
        MHF rank to use. Can be a single integer for all layers, or a dict
        specifying rank per layer name.
    factorization : str, default="tucker"
        Tensor factorization type to use
    implementation : str, default="factorized"
        MHF implementation mode
        
    Returns
    -------
    nn.Module
        MHF-compressed model with the same API as the original
    """
    # This is a generic implementation that walks the model
    # and replaces SpectralConv layers with SpectralConvMHF
    def replace_spectral_conv(
        module: nn.Module,
        path: str = "",
    ) -> nn.Module:
        """Recursively replace spectral conv layers"""
        from neuralop.layers.spectral_convolution import SpectralConv
        
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            
            if isinstance(child, SpectralConv):
                # Get rank for this layer
                if isinstance(mhf_rank, dict):
                    layer_rank = mhf_rank.get(child_path, mhf_rank.get("default", 8))
                else:
                    layer_rank = mhf_rank
                
                # Convert to MHF
                mhf_conv = SpectralConvMHF(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    n_modes=child.n_modes,
                    mhf_rank=layer_rank,
                    factorization=factorization,
                    implementation=implementation,
                    complex_data=child.complex_data if hasattr(child, "complex_data") else False,
                )
                
                # Copy original weights
                with torch.no_grad():
                    mhf_conv.weight.copy_(child.weight)
                
                # Perform decomposition
                mhf_conv.decompose()
                
                # Replace in parent
                setattr(module, name, mhf_conv)
            
            else:
                # Recurse
                replace_spectral_conv(child, child_path)
        
        return module
    
    return replace_spectral_conv(model)


def load_original_and_compress(
    checkpoint_path: str,
    model_name: str,
    model_kwargs: Dict[str, Any],
    mhf_rank: Union[int, Dict[str, Union[int, list]]],
    factorization: str = "tucker",
    output_path: Optional[str] = None,
) -> nn.Module:
    """Load an original model checkpoint, compress it with MHF, and optionally save
    
    Parameters
    ----------
    checkpoint_path : str
        Path to original model checkpoint
    model_name : str
        Name of the original model
    model_kwargs : Dict[str, Any]
        Model configuration for the original model
    mhf_rank : Union[int, Dict]
        MHF rank(s) to use for compression
    factorization : str, default="tucker"
        Factorization type
    output_path : str, optional
        Path to save the compressed model
        
    Returns
    -------
    nn.Module
        Compressed MHF model
    """
    # Load original model
    original_model = get_model(model_name, **model_kwargs)
    checkpoint = torch.load(checkpoint_path)
    
    if "model_state_dict" in checkpoint:
        original_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        original_model.load_state_dict(checkpoint)
    
    # Compress
    compressed_model = compress_pretrained_model(
        original_model,
        mhf_rank=mhf_rank,
        factorization=factorization,
    )
    
    # Save if requested
    if output_path is not None:
        torch.save({
            "model_state_dict": compressed_model.state_dict(),
            "mhf_rank": mhf_rank,
            "factorization": factorization,
            "original_model_name": model_name,
            "original_model_kwargs": model_kwargs,
        }, output_path)
    
    return compressed_model


def list_available_models() -> Dict[str, Type[nn.Module]]:
    """List all available MHF models
    
    Returns
    -------
    Dict[str, Type[nn.Module]]
        Dictionary mapping model names to model classes
    """
    return {k: v for k, v in _MODEL_REGISTRY.items() if not (k.startswith("mhf") and "-" in k)}
