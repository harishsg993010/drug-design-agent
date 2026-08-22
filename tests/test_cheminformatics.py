"""
Tests for Cheminformatics Modules
"""

import pytest
from unittest.mock import patch, MagicMock
from drug_discovery_mcp.cheminformatics import (
    DescriptorCalculator,
    ConversionTools,
    SimilarityTools,
    FingerprintTools,
    ADMETPredictor,
    DrugLikenessChecker,
    CheminformaticsTools,
    CheminformaticsError,
    calculate_descriptors,
    smiles_to_inchi,
    inchi_to_smiles,
    molecular_similarity,
    calculate_fingerprint,
    predict_admet,
    check_drug_likeness
)


class TestDescriptorCalculator:
    """Tests for DescriptorCalculator"""
    
    @pytest.fixture
    def calculator(self):
        return DescriptorCalculator()
    
    def test_initialization(self, calculator):
        """Test calculator initialization"""
        assert calculator is not None
    
    def test_calculate(self, calculator):
        """Test calculating descriptors"""
        # Test with a simple molecule
        smiles = "CCO"  # Ethanol
        
        try:
            result = calculator.calculate(smiles)
            
            assert result is not None
            assert result.smiles == "CCO"
            assert result.molecular_weight > 0
            assert result.logp is not None
            assert result.hba >= 0
            assert result.hbd >= 0
            
        except CheminformaticsError as e:
            # If RDKit is not available, this is expected
            assert "RDKit" in str(e)
    
    def test_calculate_batch(self, calculator):
        """Test calculating descriptors for multiple molecules"""
        smiles_list = ["CCO", "CCO", "CCO"]
        
        try:
            results = calculator.calculate_batch(smiles_list)
            
            assert results is not None
            assert len(results) == 3
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)


class TestConversionTools:
    """Tests for ConversionTools"""
    
    @pytest.fixture
    def converter(self):
        return ConversionTools()
    
    def test_initialization(self, converter):
        """Test converter initialization"""
        assert converter is not None
    
    def test_smiles_to_inchi(self, converter):
        """Test converting SMILES to InChI"""
        try:
            result = converter.smiles_to_inchi("CCO")
            
            assert result is not None
            assert result.success is True
            assert result.output_format == "InChI"
            assert len(result.output_value) > 0
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)
    
    def test_inchi_to_smiles(self, converter):
        """Test converting InChI to SMILES"""
        # Use a known InChI for ethanol
        inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        
        try:
            result = converter.inchi_to_smiles(inchi)
            
            assert result is not None
            assert result.success is True
            assert result.output_format == "SMILES"
            assert len(result.output_value) > 0
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)


class TestSimilarityTools:
    """Tests for SimilarityTools"""
    
    @pytest.fixture
    def similarity(self):
        return SimilarityTools()
    
    def test_initialization(self, similarity):
        """Test similarity tools initialization"""
        assert similarity is not None
    
    def test_calculate(self, similarity):
        """Test calculating molecular similarity"""
        try:
            result = similarity.calculate("CCO", "CCO")
            
            assert result is not None
            assert result.smiles1 == "CCO"
            assert result.smiles2 == "CCO"
            assert result.similarity == 1.0  # Same molecule should have similarity of 1
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)
    
    def test_calculate_different(self, similarity):
        """Test calculating similarity between different molecules"""
        try:
            result = similarity.calculate("CCO", "CC")
            
            assert result is not None
            assert result.similarity >= 0.0
            assert result.similarity <= 1.0
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)


class TestFingerprintTools:
    """Tests for FingerprintTools"""
    
    @pytest.fixture
    def fp_tools(self):
        return FingerprintTools()
    
    def test_initialization(self, fp_tools):
        """Test fingerprint tools initialization"""
        assert fp_tools is not None
    
    def test_calculate(self, fp_tools):
        """Test calculating fingerprint"""
        try:
            result = fp_tools.calculate("CCO", fingerprint_type="morgan")
            
            assert result is not None
            assert result.smiles == "CCO"
            assert result.fingerprint_type == "morgan"
            assert len(result.fingerprint) > 0
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)


class TestADMETPredictor:
    """Tests for ADMETPredictor"""
    
    @pytest.fixture
    def predictor(self):
        return ADMETPredictor()
    
    def test_initialization(self, predictor):
        """Test predictor initialization"""
        assert predictor is not None
    
    def test_predict(self, predictor):
        """Test predicting ADMET properties"""
        try:
            result = predictor.predict("CCO")
            
            assert result is not None
            assert result.smiles == "CCO"
            assert result.human_intestinal_absorption is not None
            assert result.caco2_permeability is not None
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)


class TestDrugLikenessChecker:
    """Tests for DrugLikenessChecker"""
    
    @pytest.fixture
    def checker(self):
        return DrugLikenessChecker()
    
    def test_initialization(self, checker):
        """Test checker initialization"""
        assert checker is not None
    
    def test_check(self, checker):
        """Test checking drug-likeness"""
        try:
            result = checker.check("CCO")
            
            assert result is not None
            assert result.smiles == "CCO"
            assert result.lipinski is not None
            assert result.ghose is not None
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)


class TestCheminformaticsTools:
    """Tests for CheminformaticsTools"""
    
    @pytest.fixture
    def tools(self):
        return CheminformaticsTools()
    
    def test_initialization(self, tools):
        """Test tools initialization"""
        assert tools is not None
        assert hasattr(tools, 'descriptor_calculator')
        assert hasattr(tools, 'conversion_tools')
        assert hasattr(tools, 'similarity_tools')
    
    def test_calculate_descriptors(self, tools):
        """Test calculating descriptors through tools"""
        try:
            result = tools.calculate_descriptors("CCO")
            
            assert result is not None
            assert "smiles" in result
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)


class TestCheminformaticsError:
    """Tests for CheminformaticsError"""
    
    def test_error_creation(self):
        """Test creating a CheminformaticsError"""
        error = CheminformaticsError("Test error", details={"smiles": "CCO"})
        
        assert error.message == "Test error"
        assert error.details == {"smiles": "CCO"}
        assert "Test error" in str(error)


# Convenience function tests

class TestConvenienceFunctions:
    """Tests for convenience functions"""
    
    def test_calculate_descriptors(self):
        """Test calculate_descriptors function"""
        try:
            result = calculate_descriptors("CCO")
            
            assert result is not None
            assert "smiles" in result
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)
    
    def test_smiles_to_inchi(self):
        """Test smiles_to_inchi function"""
        try:
            result = smiles_to_inchi("CCO")
            
            assert result is not None
            assert "output" in result
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)
    
    def test_molecular_similarity(self):
        """Test molecular_similarity function"""
        try:
            result = molecular_similarity("CCO", "CCO")
            
            assert result is not None
            assert "similarity" in result
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)
    
    def test_calculate_fingerprint(self):
        """Test calculate_fingerprint function"""
        try:
            result = calculate_fingerprint("CCO")
            
            assert result is not None
            assert "fingerprint" in result
            
        except CheminformaticsError as e:
            assert "RDKit" in str(e)
