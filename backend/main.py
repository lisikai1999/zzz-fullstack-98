"""FastAPI application for four-bar linkage kinematics simulation."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from kinematics import (
    LinkageParams, grashof_condition, solve_position,
    compute_full_rotation, find_dead_points, find_extreme_positions,
    synthesize_coupler_curve
)
from database import init_db, get_presets, get_preset, save_preset, get_trajectories, save_trajectory

app = FastAPI(title="四杆机构运动学仿真", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


class LinkageInput(BaseModel):
    a: float
    b: float
    c: float
    d: float
    coupler_x: float = 0.0
    coupler_y: float = 0.0


class PositionInput(LinkageInput):
    theta2_deg: float
    branch: int = 1


class SynthesisInput(BaseModel):
    target_points: list
    iterations: int = 1000


class PresetInput(BaseModel):
    name: str
    category: str = "custom"
    a: float
    b: float
    c: float
    d: float
    coupler_x: float = 0.0
    coupler_y: float = 0.0
    description: str = ""


class TrajectoryInput(BaseModel):
    name: str
    points: list
    description: str = ""


@app.on_event("startup")
def startup():
    init_db()


@app.get("/api/grashof")
def api_grashof(a: float, b: float, c: float, d: float):
    return grashof_condition(a, b, c, d)


@app.post("/api/position")
def api_position(data: PositionInput):
    import numpy as np
    params = LinkageParams(a=data.a, b=data.b, c=data.c, d=data.d,
                           coupler_x=data.coupler_x, coupler_y=data.coupler_y)
    theta2 = np.radians(data.theta2_deg)
    state = solve_position(params, theta2, data.branch)
    if not state.valid:
        raise HTTPException(status_code=400, detail="无法装配：杆长不满足三角条件")
    return {
        "theta2_deg": data.theta2_deg,
        "theta3_deg": float(np.degrees(state.theta3)),
        "theta4_deg": float(np.degrees(state.theta4)),
        "A": list(state.A),
        "B": list(state.B),
        "C": list(state.C),
        "D": list(state.D),
        "P": list(state.P),
        "mu_deg": float(np.degrees(state.mu)),
    }


@app.post("/api/full_rotation")
def api_full_rotation(data: LinkageInput):
    params = LinkageParams(a=data.a, b=data.b, c=data.c, d=data.d,
                           coupler_x=data.coupler_x, coupler_y=data.coupler_y)
    result = compute_full_rotation(params, steps=360)
    result["grashof"] = grashof_condition(data.a, data.b, data.c, data.d)
    result["dead_points"] = find_dead_points(params)
    result["extreme_positions"] = find_extreme_positions(params)
    return result


@app.post("/api/synthesis")
def api_synthesis(data: SynthesisInput):
    if len(data.target_points) < 5:
        raise HTTPException(status_code=400, detail="至少需要5个目标点")
    result = synthesize_coupler_curve(data.target_points, iterations=data.iterations)
    return result


@app.get("/api/presets")
def api_presets():
    return get_presets()


@app.get("/api/presets/{preset_id}")
def api_preset(preset_id: int):
    p = get_preset(preset_id)
    if not p:
        raise HTTPException(status_code=404, detail="预设不存在")
    return p


@app.post("/api/presets")
def api_save_preset(data: PresetInput):
    save_preset(data.name, data.category, data.a, data.b, data.c, data.d,
                data.coupler_x, data.coupler_y, data.description)
    return {"status": "ok"}


@app.get("/api/trajectories")
def api_trajectories():
    return get_trajectories()


@app.post("/api/trajectories")
def api_save_trajectory(data: TrajectoryInput):
    save_trajectory(data.name, data.points, data.description)
    return {"status": "ok"}


# Serve frontend
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
