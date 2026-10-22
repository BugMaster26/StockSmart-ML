def test_inventory_alert_public_functions_import_from_module():
    from src.inventory_alerts import (
        alerts_to_csv,
        create_inventory_alerts,
        create_observed_inventory_alerts,
    )

    assert callable(alerts_to_csv)
    assert callable(create_inventory_alerts)
    assert callable(create_observed_inventory_alerts)
