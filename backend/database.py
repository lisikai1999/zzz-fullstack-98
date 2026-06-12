"""SQLite database for mechanism presets and trajectory library."""

import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "linkage.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS presets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'custom',
            a REAL NOT NULL,
            b REAL NOT NULL,
            c REAL NOT NULL,
            d REAL NOT NULL,
            coupler_x REAL NOT NULL DEFAULT 0,
            coupler_y REAL NOT NULL DEFAULT 0,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS trajectory_library (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            points TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Insert default presets if empty
    count = conn.execute("SELECT COUNT(*) FROM presets").fetchone()[0]
    if count == 0:
        defaults = [
            ("经典曲柄摇杆", "crank_rocker", 1, 3.5, 3, 4, 1.8, 0.8, "a最短且为曲柄，满足Grashof条件"),
            ("双曲柄机构", "double_crank", 3, 4, 3.5, 1.5, 2, 0.5, "d最短(机架)，两侧杆均可整转"),
            ("双摇杆机构", "double_rocker", 4, 5, 4, 7, 2.5, 1, "s+l>p+q，不满足Grashof条件"),
            ("平行四边形", "special", 3, 5, 3, 5, 2.5, 1, "对边等长a=c,b=d的特殊机构"),
            ("等腰梯形", "special", 3, 5, 3, 4, 2.5, 1.5, "两侧杆等长a=c的反平行四边形"),
            ("急回机构", "crank_rocker", 1, 4, 3.5, 4.5, 2, 1.2, "行程速比系数K>1的曲柄摇杆"),
        ]
        conn.executemany(
            "INSERT INTO presets (name, category, a, b, c, d, coupler_x, coupler_y, description) VALUES (?,?,?,?,?,?,?,?,?)",
            defaults
        )

    # Insert default target trajectories
    traj_count = conn.execute("SELECT COUNT(*) FROM trajectory_library").fetchone()[0]
    if traj_count == 0:
        import numpy as np
        # Circle
        t = np.linspace(0, 2*np.pi, 30, endpoint=False)
        circle = [[float(2*np.cos(a)+5), float(2*np.sin(a)+3)] for a in t]

        # Figure-eight
        eight = [[float(2*np.sin(2*a)+5), float(2*np.sin(a)+3)] for a in t]

        # Ellipse
        ellipse = [[float(3*np.cos(a)+5), float(1.5*np.sin(a)+3)] for a in t]

        # Straight line segment approximation
        line = [[float(x), 3.0] for x in np.linspace(2, 7, 30)]

        trajectories = [
            ("圆形轨迹", json.dumps(circle), "连杆点近似画圆"),
            ("8字形轨迹", json.dumps(eight), "典型连杆曲线形状"),
            ("椭圆轨迹", json.dumps(ellipse), "椭圆近似轨迹"),
            ("近似直线", json.dumps(line), "连杆点近似直线运动"),
        ]
        conn.executemany(
            "INSERT INTO trajectory_library (name, points, description) VALUES (?,?,?)",
            trajectories
        )

    conn.commit()
    conn.close()


def get_presets():
    conn = get_db()
    rows = conn.execute("SELECT * FROM presets ORDER BY category, name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_preset(preset_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM presets WHERE id=?", (preset_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_preset(name, category, a, b, c, d, coupler_x, coupler_y, description=""):
    conn = get_db()
    conn.execute(
        "INSERT INTO presets (name, category, a, b, c, d, coupler_x, coupler_y, description) VALUES (?,?,?,?,?,?,?,?,?)",
        (name, category, a, b, c, d, coupler_x, coupler_y, description)
    )
    conn.commit()
    conn.close()


def get_trajectories():
    conn = get_db()
    rows = conn.execute("SELECT * FROM trajectory_library ORDER BY name").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["points"] = json.loads(d["points"])
        result.append(d)
    return result


def save_trajectory(name, points, description=""):
    conn = get_db()
    conn.execute(
        "INSERT INTO trajectory_library (name, points, description) VALUES (?,?,?)",
        (name, json.dumps(points), description)
    )
    conn.commit()
    conn.close()
