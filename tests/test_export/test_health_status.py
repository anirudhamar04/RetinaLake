"""Tests for the cross-dataset normal/abnormal health_status export field."""

from chaksudb.export.query_builder import QueryBuilder
from chaksudb.export.spec import ExportSpec


def _sql(**kw) -> str:
    return QueryBuilder().build_query(ExportSpec(**kw)).render_sql()


def test_health_status_column_added_when_requested():
    sql = _sql(include_health_status=True)
    assert "AS health_status" in sql
    # derives from both grading and classification
    assert "disease_grading" in sql and "classification_annotations" in sql
    # verdict logic present
    assert "'abnormal'" in sql and "'normal'" in sql


def test_no_health_status_by_default():
    assert "AS health_status" not in _sql(dataset_names=["FIVES"])


def test_normal_filter_applies_where_and_implies_column():
    sql = _sql(dataset_names=["FIVES"], health_status_filter="normal")
    assert "health.health_status = " in sql        # filter
    assert "AS health_status" in sql                # implied column


def test_abnormal_filter():
    sql = _sql(health_status_filter="abnormal")
    assert "health.health_status = " in sql


def test_abnormal_evidence_uses_normalized_grade_and_concepts():
    sql = _sql(include_health_status=True)
    assert "dg.scaled_grade >= 1" in sql                       # any disease grade
    assert "c.concept <> 'normal'" in sql                      # any disease concept positive
    assert "c.task_type = 'multi_class'" in sql                # multi_class winner counts
