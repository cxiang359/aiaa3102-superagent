#!/usr/bin/env python
"""Unified execution entry point for the General Algorithm SuperAgent.

Example:
    python run_agent.py --task-dir "Released_Task1-In_Vehicle_Coupon_Recommendation/in_vehicle_coupon_recommendation" \
                        --task-name task1_coupon --metric accuracy
"""

import sys

from superagent.main import main

if __name__ == "__main__":
    sys.exit(main())
