"""
OpenTargets Database Client

Provides access to OpenTargets platform for target validation and disease association data.

The OpenTargets Platform is a GraphQL API: every operation below is expressed as a
GraphQL document POSTed to a single endpoint, rather than as a REST path.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base import DatabaseClient, DatabaseConfig, DatabaseError

logger = logging.getLogger(__name__)


@dataclass
class Target:
    """Represents a target from OpenTargets"""
    id: str
    name: str
    gene_symbol: str
    gene_id: Optional[str] = None
    protein_id: Optional[str] = None
    target_type: str = "PROTEIN"

    # Additional metadata
    description: Optional[str] = None
    synonyms: Optional[List[str]] = None

    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []


@dataclass
class Disease:
    """Represents a disease from OpenTargets"""
    id: str
    name: str
    description: Optional[str] = None

    # Classification
    disease_type: Optional[str] = None
    therapeutic_areas: Optional[List[str]] = None

    def __post_init__(self):
        if self.therapeutic_areas is None:
            self.therapeutic_areas = []


@dataclass
class Association:
    """Represents a target-disease association"""
    target_id: str
    disease_id: str
    score: float

    # Evidence
    evidence: Optional[List[Dict[str, Any]]] = None

    # Additional metadata
    mechanism: Optional[str] = None
    action_type: Optional[str] = None

    # Human-readable labels for the two ends of the association
    target_symbol: Optional[str] = None
    disease_name: Optional[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


# --- GraphQL documents -------------------------------------------------------

_TARGET_FIELDS = """
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    synonyms { label source }
    proteinIds { id source }
"""

_QUERY_TARGET = """
query Target($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    %s
  }
}
""" % _TARGET_FIELDS

_SEARCH = """
query Search($queryString: String!, $entityNames: [String!], $index: Int!, $size: Int!) {
  search(queryString: $queryString, entityNames: $entityNames,
         page: {index: $index, size: $size}) {
    total
    hits { id name entity description }
  }
}
"""

_QUERY_DISEASE = """
query DiseaseQuery($efoId: String!) {
  disease(efoId: $efoId) {
    id
    name
    description
    therapeuticAreas { id name }
    synonyms { relation terms }
  }
}
"""

_TARGET_ASSOCIATIONS = """
query TargetAssociations($ensemblId: String!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(page: {index: 0, size: $size}, orderByScore: "score") {
      count
      rows {
        score
        datatypeScores { id score }
        disease { id name }
      }
    }
  }
}
"""

_DISEASE_ASSOCIATIONS = """
query DiseaseAssociations($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: 0, size: $size}, orderByScore: "score") {
      count
      rows {
        score
        datatypeScores { id score }
        target { id approvedSymbol approvedName }
      }
    }
  }
}
"""

_EVIDENCE = """
query Evidence($ensemblId: String!, $efoIds: [String!]!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    evidences(efoIds: $efoIds, size: $size) {
      count
      rows {
        id
        score
        datasourceId
        datatypeId
        disease { id name }
      }
    }
  }
}
"""

_DRUGS = """
query Drugs($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug {
          id
          name
          drugType
          description
          maximumClinicalStage
          tradeNames { label }
          synonyms { label }
          mechanismsOfAction { rows { mechanismOfAction actionType } }
        }
        diseases { disease { id name } }
        clinicalReports {
          id
          trialPhase
          trialOverallStatus
          trialStartDate
          url
          source
        }
      }
    }
  }
}
"""


class OpenTargetsClient(DatabaseClient):
    """
    Client for querying OpenTargets platform

    OpenTargets is a platform for target validation and disease association analysis.
    It provides:
    - Target information
    - Disease information
    - Target-disease associations
    - Evidence scoring
    - Drug information
    - Clinical trial data
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize OpenTargets client

        Args:
            config: Custom configuration
        """
        super().__init__(config or self.get_default_config())

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        """Get default OpenTargets configuration"""
        return DatabaseConfig(
            endpoint="https://api.platform.opentargets.org/api/v4/graphql",
            rate_limit=10,
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=3600,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "DrugDiscoveryMCP/0.1.0"
            }
        )

    def get_name(self) -> str:
        """Get the name of this database"""
        return "OpenTargets"

    def _graphql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL document and return its ``data`` payload

        GraphQL reports failures with HTTP 200 and an ``errors`` array, so the
        transport-level check in ``_make_request`` is not enough on its own.

        Args:
            query: GraphQL document
            variables: Variable bindings for the document

        Returns:
            The ``data`` object from the response

        Raises:
            DatabaseError: If the API reports GraphQL errors
        """
        payload = {"query": query, "variables": variables or {}}
        response = self._make_request("POST", self.config.endpoint, data=payload)

        if response.get("errors"):
            messages = "; ".join(
                str(err.get("message", err)) for err in response["errors"]
            )
            raise DatabaseError(
                message=f"GraphQL error: {messages}",
                details={"errors": response["errors"]},
            )

        data = response.get("data")
        if data is None:
            raise DatabaseError(
                message="GraphQL response contained no data",
                details={"response": response},
            )
        return data

    def query_target(self, target_id: str) -> Target:
        """
        Query OpenTargets for a specific target

        Args:
            target_id: Ensembl gene ID (e.g., "ENSG00000141510")

        Returns:
            Target object with target information
        """
        try:
            data = self._graphql(_QUERY_TARGET, {"ensemblId": target_id})
            target = data.get("target")
            if not target:
                raise DatabaseError(f"Target not found: {target_id}")
            return self._parse_target(target)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"OpenTargets target query failed for {target_id}: {e}")
            raise DatabaseError(f"Failed to query OpenTargets target {target_id}: {e}")

    def search_targets(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        gene_symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search OpenTargets targets

        Args:
            query: Search query
            limit: Maximum number of results
            offset: Pagination offset (rounded down to a whole page)
            gene_symbol: Restrict the search to this gene symbol

        Returns:
            Dictionary with search results
        """
        return self._search(query, "target", limit, offset, override_query=gene_symbol)

    def query_disease(self, disease_id: str) -> Disease:
        """
        Query OpenTargets for a specific disease

        Args:
            disease_id: EFO/MONDO disease ID (e.g., "MONDO_0004992")

        Returns:
            Disease object with disease information
        """
        try:
            data = self._graphql(_QUERY_DISEASE, {"efoId": disease_id})
            disease = data.get("disease")
            if not disease:
                raise DatabaseError(f"Disease not found: {disease_id}")
            return self._parse_disease(disease)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"OpenTargets disease query failed for {disease_id}: {e}")
            raise DatabaseError(f"Failed to query OpenTargets disease {disease_id}: {e}")

    def search_diseases(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Search OpenTargets diseases

        Args:
            query: Search query
            limit: Maximum number of results
            offset: Pagination offset (rounded down to a whole page)

        Returns:
            Dictionary with search results
        """
        return self._search(query, "disease", limit, offset)

    def _search(
        self,
        query: str,
        entity: str,
        limit: int,
        offset: int,
        override_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the shared search query for a single entity type"""
        query_string = override_query or query
        size = max(1, limit)
        index = offset // size if offset else 0

        try:
            data = self._graphql(_SEARCH, {
                "queryString": query_string,
                "entityNames": [entity],
                "index": index,
                "size": size,
            })
            search = data.get("search") or {}
            parser = self._parse_target_summary if entity == "target" else self._parse_disease_summary
            return {
                "results": [parser(hit) for hit in search.get("hits", [])],
                "total": search.get("total", 0),
                "limit": limit,
                "offset": offset,
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"OpenTargets {entity} search failed: {e}")
            raise DatabaseError(f"Failed to search OpenTargets {entity}s: {e}")

    def get_associations(
        self,
        target_id: Optional[str] = None,
        disease_id: Optional[str] = None,
        limit: int = 100,
        score_threshold: float = 0.1
    ) -> List[Association]:
        """
        Get target-disease associations

        Exactly one of ``target_id`` or ``disease_id`` drives the query; if both
        are given the target is used and the results are filtered to the disease.

        Args:
            target_id: Ensembl gene ID
            disease_id: EFO/MONDO disease ID
            limit: Maximum number of results
            score_threshold: Minimum association score

        Returns:
            List of Association objects
        """
        if not target_id and not disease_id:
            raise DatabaseError("get_associations requires target_id or disease_id")

        try:
            if target_id:
                data = self._graphql(_TARGET_ASSOCIATIONS, {
                    "ensemblId": target_id,
                    "size": limit,
                })
                target = data.get("target") or {}
                block = target.get("associatedDiseases") or {}
                associations = [
                    Association(
                        target_id=target.get("id", target_id),
                        target_symbol=target.get("approvedSymbol"),
                        disease_id=(row.get("disease") or {}).get("id", ""),
                        disease_name=(row.get("disease") or {}).get("name"),
                        score=row.get("score", 0.0) or 0.0,
                        evidence=row.get("datatypeScores", []) or [],
                    )
                    for row in block.get("rows", [])
                ]
                if disease_id:
                    associations = [a for a in associations if a.disease_id == disease_id]
            else:
                data = self._graphql(_DISEASE_ASSOCIATIONS, {
                    "efoId": disease_id,
                    "size": limit,
                })
                disease = data.get("disease") or {}
                block = disease.get("associatedTargets") or {}
                associations = [
                    Association(
                        target_id=(row.get("target") or {}).get("id", ""),
                        target_symbol=(row.get("target") or {}).get("approvedSymbol"),
                        disease_id=disease.get("id", disease_id),
                        disease_name=disease.get("name"),
                        score=row.get("score", 0.0) or 0.0,
                        evidence=row.get("datatypeScores", []) or [],
                    )
                    for row in block.get("rows", [])
                ]

            return [a for a in associations if a.score >= score_threshold]

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"OpenTargets association query failed: {e}")
            raise DatabaseError(f"Failed to query OpenTargets associations: {e}")

    def get_target_diseases(self, target_id: str) -> List[Association]:
        """
        Get diseases associated with a specific target

        Args:
            target_id: Ensembl gene ID

        Returns:
            List of Association objects
        """
        return self.get_associations(target_id=target_id)

    def get_disease_targets(self, disease_id: str) -> List[Association]:
        """
        Get targets associated with a specific disease

        Args:
            disease_id: EFO/MONDO disease ID

        Returns:
            List of Association objects
        """
        return self.get_associations(disease_id=disease_id)

    def get_evidence(
        self,
        target_id: str,
        disease_id: str,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Get evidence for a specific target-disease association

        Args:
            target_id: Ensembl gene ID
            disease_id: EFO/MONDO disease ID
            limit: Maximum number of evidence records

        Returns:
            List of evidence records
        """
        try:
            data = self._graphql(_EVIDENCE, {
                "ensemblId": target_id,
                "efoIds": [disease_id],
                "size": limit,
            })
            target = data.get("target") or {}
            block = target.get("evidences") or {}
            return [
                {
                    "id": row.get("id"),
                    "score": row.get("score"),
                    "datasource": row.get("datasourceId"),
                    "datatype": row.get("datatypeId"),
                    "disease_id": (row.get("disease") or {}).get("id"),
                    "disease_name": (row.get("disease") or {}).get("name"),
                }
                for row in block.get("rows", [])
            ]
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"OpenTargets evidence query failed: {e}")
            raise DatabaseError(f"Failed to query OpenTargets evidence: {e}")

    def get_drugs(self, target_id: str) -> List[Dict[str, Any]]:
        """
        Get drugs and clinical candidates associated with a target

        Args:
            target_id: Ensembl gene ID

        Returns:
            List of drug information
        """
        try:
            return [self._parse_drug_row(row) for row in self._drug_rows(target_id)]
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"OpenTargets drugs query failed: {e}")
            raise DatabaseError(f"Failed to query OpenTargets drugs: {e}")

    def get_clinical_trials(self, target_id: str) -> List[Dict[str, Any]]:
        """
        Get clinical trials associated with a target

        Args:
            target_id: Ensembl gene ID

        Returns:
            List of clinical trial information
        """
        try:
            trials = []
            for row in self._drug_rows(target_id):
                drug = row.get("drug") or {}
                for report in row.get("clinicalReports") or []:
                    trials.append({
                        "trial_id": report.get("id"),
                        "phase": report.get("trialPhase"),
                        "status": report.get("trialOverallStatus"),
                        "start_date": report.get("trialStartDate"),
                        "url": report.get("url"),
                        "source": report.get("source"),
                        "drug_id": drug.get("id"),
                        "drug_name": drug.get("name"),
                    })
            return trials
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"OpenTargets clinical trials query failed: {e}")
            raise DatabaseError(f"Failed to query OpenTargets clinical trials: {e}")

    def _drug_rows(self, target_id: str) -> List[Dict[str, Any]]:
        """Fetch the raw drug/clinical-candidate rows for a target"""
        data = self._graphql(_DRUGS, {"ensemblId": target_id})
        target = data.get("target") or {}
        block = target.get("drugAndClinicalCandidates") or {}
        return block.get("rows", []) or []

    # --- parsers -------------------------------------------------------------

    @staticmethod
    def _labels(items: Optional[List[Dict[str, Any]]]) -> List[str]:
        """Collect the ``label`` of each entry, dropping blanks and duplicates"""
        seen = []
        for item in items or []:
            label = (item or {}).get("label")
            if label and label not in seen:
                seen.append(label)
        return seen

    def _parse_target(self, data: Dict[str, Any]) -> Target:
        """Parse target data into Target object"""
        protein_ids = data.get("proteinIds") or []
        uniprot = next(
            (p.get("id") for p in protein_ids
             if str(p.get("source", "")).startswith("uniprot")),
            None,
        )
        descriptions = data.get("functionDescriptions") or []

        return Target(
            id=data.get("id", ""),
            name=data.get("approvedName", ""),
            gene_symbol=data.get("approvedSymbol", ""),
            gene_id=data.get("id"),
            protein_id=uniprot or (protein_ids[0].get("id") if protein_ids else None),
            target_type=data.get("biotype") or "PROTEIN",
            description=descriptions[0] if descriptions else None,
            synonyms=self._labels(data.get("synonyms")),
        )

    def _parse_target_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a target search hit"""
        return {
            "id": data.get("id", ""),
            "gene_symbol": data.get("name", ""),
            "description": data.get("description"),
            "entity": data.get("entity", "target"),
        }

    def _parse_disease(self, data: Dict[str, Any]) -> Disease:
        """Parse disease data into Disease object"""
        areas = [a.get("name") for a in data.get("therapeuticAreas") or [] if a.get("name")]

        return Disease(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            disease_type=areas[0] if areas else None,
            therapeutic_areas=areas,
        )

    def _parse_disease_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a disease search hit"""
        return {
            "id": data.get("id", ""),
            "name": data.get("name", ""),
            "description": data.get("description"),
            "entity": data.get("entity", "disease"),
        }

    def _parse_drug_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a drug/clinical-candidate row"""
        drug = row.get("drug") or {}
        mechanisms = ((drug.get("mechanismsOfAction") or {}).get("rows")) or []
        diseases = [
            (d.get("disease") or {})
            for d in row.get("diseases") or []
        ]

        return {
            "id": drug.get("id"),
            "name": drug.get("name"),
            "drug_type": drug.get("drugType"),
            "description": drug.get("description"),
            "max_clinical_phase": drug.get("maximumClinicalStage") or row.get("maxClinicalStage"),
            "trade_names": self._labels(drug.get("tradeNames")),
            "synonyms": self._labels(drug.get("synonyms")),
            "mechanisms_of_action": [
                {
                    "mechanism": m.get("mechanismOfAction"),
                    "action_type": m.get("actionType"),
                }
                for m in mechanisms
            ],
            "diseases": [
                {"id": d.get("id"), "name": d.get("name")}
                for d in diseases if d.get("id")
            ],
            "trial_count": len(row.get("clinicalReports") or []),
        }

    def _parse_association(self, data: Dict[str, Any]) -> Association:
        """Parse an association row into an Association object"""
        target = data.get("target") or {}
        disease = data.get("disease") or {}

        return Association(
            target_id=target.get("id", ""),
            target_symbol=target.get("approvedSymbol"),
            disease_id=disease.get("id", ""),
            disease_name=disease.get("name"),
            score=data.get("score", 0.0) or 0.0,
            evidence=data.get("datatypeScores", []) or [],
        )


# Singleton instance
opentargets_client = OpenTargetsClient()


# Convenience functions for direct use
def query_opentargets(target_id: str, **kwargs) -> Dict[str, Any]:
    """Query OpenTargets database"""
    try:
        target = opentargets_client.query_target(target_id, **kwargs)
        return target.__dict__
    except DatabaseError as e:
        return {"error": str(e), "target_id": target_id}


def search_opentargets(query: str, **kwargs) -> Dict[str, Any]:
    """Search OpenTargets database"""
    try:
        return opentargets_client.search_targets(query, **kwargs)
    except DatabaseError as e:
        return {"error": str(e), "query": query}
