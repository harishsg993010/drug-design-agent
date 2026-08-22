"""
Molecular Format Conversion Tools

Provides conversion between different molecular representations:
- SMILES <-> InChI
- SMILES <-> InChIKey
- SMILES <-> MOL
- SMILES <-> SDF
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base import CheminformaticsBase, CheminformaticsError

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of a molecular format conversion"""
    input_format: str
    input_value: str
    output_format: str
    output_value: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConversionTools(CheminformaticsBase):
    """
    Tools for converting between different molecular representations
    
    Supports conversion between:
    - SMILES (Simplified Molecular Input Line Entry System)
    - InChI (IUPAC International Chemical Identifier)
    - InChIKey (Hashed version of InChI)
    - MOL format
    - SDF format
    """
    
    def __init__(self):
        """Initialize conversion tools"""
        super().__init__()
    
    def smiles_to_inchi(self, smiles: str, options: Optional[Dict[str, Any]] = None) -> ConversionResult:
        """
        Convert SMILES to InChI
        
        Args:
            smiles: SMILES string
            options: Conversion options (e.g., {"fixedH": True})
            
        Returns:
            ConversionResult with InChI string
        """
        try:
            Chem = self._get_rdkit()
            
            # Sanitize SMILES
            smiles = self._sanitize_smiles(smiles)
            mol = self._smiles_to_mol(smiles)
            
            # Generate InChI
            inchi = Chem.MolToInchi(mol, options=options)
            
            if not inchi:
                raise CheminformaticsError(f"Failed to convert SMILES to InChI: {smiles}")
            
            return ConversionResult(
                input_format="SMILES",
                input_value=smiles,
                output_format="InChI",
                output_value=inchi,
                success=True
            )
            
        except Exception as e:
            logger.error(f"SMILES to InChI conversion failed: {e}")
            return ConversionResult(
                input_format="SMILES",
                input_value=smiles,
                output_format="InChI",
                output_value="",
                success=False,
                error=str(e)
            )
    
    def inchi_to_smiles(self, inchi: str, sanitize: bool = True) -> ConversionResult:
        """
        Convert InChI to SMILES
        
        Args:
            inchi: InChI string
            sanitize: Whether to sanitize the resulting SMILES
            
        Returns:
            ConversionResult with SMILES string
        """
        try:
            Chem = self._get_rdkit()
            
            # Convert InChI to molecule
            mol = Chem.MolFromInchi(inchi)
            
            if mol is None:
                raise CheminformaticsError(f"Failed to convert InChI to molecule: {inchi}")
            
            # Convert to SMILES
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            
            if sanitize:
                smiles = self._sanitize_smiles(smiles)
            
            if not smiles:
                raise CheminformaticsError(f"Failed to convert InChI to SMILES: {inchi}")
            
            return ConversionResult(
                input_format="InChI",
                input_value=inchi,
                output_format="SMILES",
                output_value=smiles,
                success=True
            )
            
        except Exception as e:
            logger.error(f"InChI to SMILES conversion failed: {e}")
            return ConversionResult(
                input_format="InChI",
                input_value=inchi,
                output_format="SMILES",
                output_value="",
                success=False,
                error=str(e)
            )
    
    def smiles_to_inchikey(self, smiles: str) -> ConversionResult:
        """
        Convert SMILES to InChIKey
        
        Args:
            smiles: SMILES string
            
        Returns:
            ConversionResult with InChIKey string
        """
        try:
            Chem = self._get_rdkit()
            
            # Sanitize SMILES
            smiles = self._sanitize_smiles(smiles)
            mol = self._smiles_to_mol(smiles)
            
            # Generate InChIKey
            inchikey = Chem.MolToInchiKey(mol)
            
            if not inchikey:
                raise CheminformaticsError(f"Failed to convert SMILES to InChIKey: {smiles}")
            
            return ConversionResult(
                input_format="SMILES",
                input_value=smiles,
                output_format="InChIKey",
                output_value=inchikey,
                success=True
            )
            
        except Exception as e:
            logger.error(f"SMILES to InChIKey conversion failed: {e}")
            return ConversionResult(
                input_format="SMILES",
                input_value=smiles,
                output_format="InChIKey",
                output_value="",
                success=False,
                error=str(e)
            )
    
    def inchikey_to_smiles(self, inchikey: str) -> ConversionResult:
        """
        Convert InChIKey to SMILES
        
        Note: InChIKey is a hashed representation and cannot be directly
        converted back to structure. This method tries to find a matching
        molecule in a database.
        
        Args:
            inchikey: InChIKey string
            
        Returns:
            ConversionResult with SMILES string (if found)
        """
        # InChIKey cannot be directly converted back to SMILES
        # This is a limitation of the InChIKey format
        return ConversionResult(
            input_format="InChIKey",
            input_value=inchikey,
            output_format="SMILES",
            output_value="",
            success=False,
            error="InChIKey cannot be directly converted to SMILES (it's a hashed representation)"
        )
    
    def inchi_to_inchikey(self, inchi: str) -> ConversionResult:
        """
        Convert InChI to InChIKey
        
        Args:
            inchi: InChI string
            
        Returns:
            ConversionResult with InChIKey string
        """
        try:
            Chem = self._get_rdkit()
            
            # Convert InChI to molecule
            mol = Chem.MolFromInchi(inchi)
            
            if mol is None:
                raise CheminformaticsError(f"Failed to convert InChI to molecule: {inchi}")
            
            # Generate InChIKey
            inchikey = Chem.MolToInchiKey(mol)
            
            if not inchikey:
                raise CheminformaticsError(f"Failed to convert InChI to InChIKey: {inchi}")
            
            return ConversionResult(
                input_format="InChI",
                input_value=inchi,
                output_format="InChIKey",
                output_value=inchikey,
                success=True
            )
            
        except Exception as e:
            logger.error(f"InChI to InChIKey conversion failed: {e}")
            return ConversionResult(
                input_format="InChI",
                input_value=inchi,
                output_format="InChIKey",
                output_value="",
                success=False,
                error=str(e)
            )
    
    def inchikey_to_inchi(self, inchikey: str) -> ConversionResult:
        """
        Convert InChIKey to InChI
        
        Note: InChIKey is a hashed representation and cannot be directly
        converted back to InChI. This method tries to find a matching
        molecule in a database.
        
        Args:
            inchikey: InChIKey string
            
        Returns:
            ConversionResult with InChI string (if found)
        """
        # InChIKey cannot be directly converted back to InChI
        return ConversionResult(
            input_format="InChIKey",
            input_value=inchikey,
            output_format="InChI",
            output_value="",
            success=False,
            error="InChIKey cannot be directly converted to InChI (it's a hashed representation)"
        )
    
    def smiles_to_mol(self, smiles: str, format: str = "mol") -> ConversionResult:
        """
        Convert SMILES to MOL or SDF format
        
        Args:
            smiles: SMILES string
            format: Output format ("mol" or "sdf")
            
        Returns:
            ConversionResult with MOL/SDF string
        """
        try:
            Chem = self._get_rdkit()
            
            # Sanitize SMILES
            smiles = self._sanitize_smiles(smiles)
            mol = self._smiles_to_mol(smiles)
            
            # Generate MOL or SDF
            if format.lower() == "sdf":
                output = Chem.MolToMolBlock(mol)
            else:
                output = Chem.MolToMolBlock(mol)  # MOL format is same as MolBlock
            
            if not output:
                raise CheminformaticsError(f"Failed to convert SMILES to {format.upper()}: {smiles}")
            
            return ConversionResult(
                input_format="SMILES",
                input_value=smiles,
                output_format=format.upper(),
                output_value=output,
                success=True
            )
            
        except Exception as e:
            logger.error(f"SMILES to {format.upper()} conversion failed: {e}")
            return ConversionResult(
                input_format="SMILES",
                input_value=smiles,
                output_format=format.upper(),
                output_value="",
                success=False,
                error=str(e)
            )
    
    def mol_to_smiles(self, mol_block: str) -> ConversionResult:
        """
        Convert MOL or SDF format to SMILES
        
        Args:
            mol_block: MOL or SDF format string
            
        Returns:
            ConversionResult with SMILES string
        """
        try:
            Chem = self._get_rdkit()
            
            # Read MOL block
            mol = Chem.MolFromMolBlock(mol_block)
            
            if mol is None:
                raise CheminformaticsError("Failed to read MOL block")
            
            # Convert to SMILES
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            smiles = self._sanitize_smiles(smiles)
            
            if not smiles:
                raise CheminformaticsError("Failed to convert MOL to SMILES")
            
            return ConversionResult(
                input_format="MOL",
                input_value=mol_block[:50] + "..." if len(mol_block) > 50 else mol_block,
                output_format="SMILES",
                output_value=smiles,
                success=True
            )
            
        except Exception as e:
            logger.error(f"MOL to SMILES conversion failed: {e}")
            return ConversionResult(
                input_format="MOL",
                input_value=mol_block[:50] + "..." if len(mol_block) > 50 else mol_block,
                output_format="SMILES",
                output_value="",
                success=False,
                error=str(e)
            )
    
    def convert(
        self,
        input_value: str,
        input_format: str,
        output_format: str,
        **kwargs
    ) -> ConversionResult:
        """
        Generic conversion method between any supported formats
        
        Args:
            input_value: Input molecular representation
            input_format: Input format ("SMILES", "InChI", "InChIKey", "MOL", "SDF")
            output_format: Output format ("SMILES", "InChI", "InChIKey", "MOL", "SDF")
            **kwargs: Additional conversion options
            
        Returns:
            ConversionResult with converted value
        """
        input_format = input_format.upper()
        output_format = output_format.upper()
        
        # Normalize formats
        if input_format == "SDF":
            input_format = "MOL"
        if output_format == "SDF":
            output_format = "MOL"
        
        # Direct conversions
        if input_format == "SMILES" and output_format == "InChI":
            return self.smiles_to_inchi(input_value, **kwargs)
        elif input_format == "InChI" and output_format == "SMILES":
            return self.inchi_to_smiles(input_value, **kwargs)
        elif input_format == "SMILES" and output_format == "InChIKey":
            return self.smiles_to_inchikey(input_value, **kwargs)
        elif input_format == "InChI" and output_format == "InChIKey":
            return self.inchi_to_inchikey(input_value, **kwargs)
        elif input_format == "SMILES" and output_format == "MOL":
            return self.smiles_to_mol(input_value, format="mol", **kwargs)
        elif input_format == "MOL" and output_format == "SMILES":
            return self.mol_to_smiles(input_value, **kwargs)
        
        # Indirect conversions (through SMILES)
        elif input_format == "InChIKey" and output_format == "SMILES":
            return self.inchikey_to_smiles(input_value, **kwargs)
        elif input_format == "InChIKey" and output_format == "InChI":
            return self.inchikey_to_inchi(input_value, **kwargs)
        
        # Convert through SMILES
        else:
            # First convert to SMILES
            if input_format == "InChIKey":
                smiles_result = self.inchikey_to_smiles(input_value, **kwargs)
            elif input_format == "MOL":
                smiles_result = self.mol_to_smiles(input_value, **kwargs)
            else:
                raise CheminformaticsError(f"Unsupported input format: {input_format}")
            
            if not smiles_result.success:
                return smiles_result
            
            # Then convert from SMILES to target format
            if output_format == "InChI":
                return self.smiles_to_inchi(smiles_result.output_value, **kwargs)
            elif output_format == "InChIKey":
                return self.smiles_to_inchikey(smiles_result.output_value, **kwargs)
            elif output_format == "MOL":
                return self.smiles_to_mol(smiles_result.output_value, format="mol", **kwargs)
            elif output_format == "SMILES":
                return smiles_result
            else:
                raise CheminformaticsError(f"Unsupported output format: {output_format}")


# Singleton instance
_conversion_tools = ConversionTools()


def smiles_to_inchi(smiles: str, **kwargs) -> Dict[str, Any]:
    """
    Convert SMILES to InChI
    
    Args:
        smiles: SMILES string
        **kwargs: Additional options
        
    Returns:
        Dictionary with conversion result
    """
    try:
        result = _conversion_tools.smiles_to_inchi(smiles, **kwargs)
        return {
            "input": result.input_value,
            "output": result.output_value,
            "success": result.success,
            "error": result.error
        }
    except CheminformaticsError as e:
        return {"error": str(e), "input": smiles}


def inchi_to_smiles(inchi: str, **kwargs) -> Dict[str, Any]:
    """
    Convert InChI to SMILES
    
    Args:
        inchi: InChI string
        **kwargs: Additional options
        
    Returns:
        Dictionary with conversion result
    """
    try:
        result = _conversion_tools.inchi_to_smiles(inchi, **kwargs)
        return {
            "input": result.input_value,
            "output": result.output_value,
            "success": result.success,
            "error": result.error
        }
    except CheminformaticsError as e:
        return {"error": str(e), "input": inchi}
