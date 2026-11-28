"""Frozen Minimal-D entry for the registered single-frame method."""
import live_2d  # Initialize the locked geometry import paths before loading Minimal-D.

from minimal import predict as recover


def predict(frame, replies):
    return recover(frame, replies, scales=(0, 2, 5, 7))
