import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cadquery as cq
import tempfile
import zipfile
from io import BytesIO
from datetime import datetime
from gcode_generator import *

# Initialize Flask app
app = Flask(__name__)

# Enable CORS to allow Flutter app to communicate
CORS(app)

def span_around(value, mids, lo, hi):
    """
    Given a coordinate value and a sorted list of separator values (mids),
    return the [left/right] or [bottom/top] span that contains the value.
    If the value lies outside the mids range, we clip to the blister edges.
    """
    # Defaults to blister edges
    left = lo
    right = hi
    # Find the nearest mid <= value and nearest mid > value
    for m in mids:
        if m <= value and m > left:
            left = m
        if m > value and m < right:
            right = m
            break  # since mids is sorted, first one > value is the nearest
    return left, right

def check_feature_intersection(x, y, features, long_side, short_side):
    """
    Check if a point (x, y) intersects with any feature rectangle.
    Returns the maximum depth needed at this point.
    
    Args:
        x, y: coordinates in the top_plane coordinate system
        features: list of feature dictionaries with x, y, width, height, thickness
        long_side, short_side: dimensions of the blister
    
    Returns:
        float: maximum cut depth needed (positive value), or 0 if no intersection
    """
    max_depth = 0.0
    
    for feature in features:
        # Convert feature coordinates to top_plane system
        feat_x = float(feature.get('x'))
        feat_y = -float(feature.get('y'))  # Negative because of coordinate system
        feat_width = float(feature.get('width'))
        feat_height = float(feature.get('height'))
        feat_thickness = float(feature.get('thickness'))
        
        # Calculate feature bounds
        feat_left = feat_x - feat_width / 2
        feat_right = feat_x + feat_width / 2
        feat_bottom = feat_y - feat_height / 2
        feat_top = feat_y + feat_height / 2
        
        # Check if point is inside feature rectangle
        if feat_left <= x <= feat_right and feat_bottom <= y <= feat_top:
            max_depth = max(max_depth, feat_thickness)
    
    return max_depth

@app.route('/convert', methods=['POST'])
def convert_to_step():
    """
    Receives JSON data from Flutter and returns a STEP file
    Expected JSON format:
    {
        'commands': 'generate_pallet',
        'parameters': {
            'rows': row_n,
            'columns': col_n,
            'short_side': short_side,
            'long_side': long_side,
            'high_border': high_border,
            'low_border': low_border,
            'left_border': left_border,
            'right_border': right_border,
            'blister_thickness': blister_thickness,
            'pill_shape': pill_shape,
            'pill_height': pill_height,
            'pill_width': pill_width,
            'pill_thickness': pill_thickness,
            'pill_pos': [...],
            'pill_rotation': [...],
            'deleted_pills': [...],
            'features': [...]  # NEW: list of feature objects
        },
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract command and parameters from JSON
        command = data.get('commands')
        params = data.get('parameters')
        
        # Convert numeric parameters to proper types
        long_side = float(params.get('long_side'))
        short_side = float(params.get('short_side'))
        blister_thickness = float(params.get('blister_thickness'))
        rows = int(params.get('rows'))
        columns = int(params.get('columns'))
        high_border = float(params.get('high_border'))
        low_border = float(params.get('low_border'))
        left_border = float(params.get('left_border'))
        right_border = float(params.get('right_border'))
        pill_shape = params.get('pill_shape')
        pill_height = float(params.get('pill_height'))
        pill_width = float(params.get('pill_width'))
        pill_thickness = blister_thickness - float(params.get('pill_thickness'))
        pill_pos = params.get('pill_pos')
        pill_rot = params.get('pill_rotation')
        dels_pills = params.get('deleted_pills')
        features = params.get('features', [])  # NEW: Get features list

        # Handle generate_pallet command
        if command == 'generate_pallet':
            # Create a simple pallet-like box using provided dimensions
            pallet_x = 182.0
            pallet_y = 122.0
            guide_height = 16.0
            pallet_height = 23.0
            cut_width = 2.0
            cut_depth = 10.0
            slot_depth = 5.0

            piano = (cq.Workplane('XY').rect(pallet_x, pallet_y).extrude(1).translate((0,0,0)))
            guide_vert1 = (cq.Workplane('XY').rect(5, pallet_y-15).extrude(guide_height).translate((pallet_x/2 - 5/2, -15/2, 1)))
            guide_vert2 = (cq.Workplane('XY').rect(5, pallet_y-15).extrude(guide_height).translate((-pallet_x/2 + 5/2, -15/2, 1)))

            guide_orr_1 = (cq.Workplane('XY').rect(pallet_x, 5).extrude(guide_height).translate((0, pallet_y/2 - 5/2, 1)))
            guide_orr_2 = (cq.Workplane('XY').rect(pallet_x, 5).extrude(guide_height).translate((0, pallet_y/2 -15 -5/2, 1)))
            guide_orr_3 = (cq.Workplane('XY').rect(pallet_x, 5).extrude(guide_height).translate((0, pallet_y/2 -short_side +5/2, 1)))

            blister = (cq.Workplane('XY').rect(long_side, short_side).extrude(pallet_height-blister_thickness).translate((long_side/2-pallet_x/2,-short_side/2+pallet_y/2,1)))

            result = piano.union(guide_vert1).union(guide_vert2).union(guide_orr_1).union(guide_orr_2).union(guide_orr_3).union(blister)
            points = []

            # Build points from pill positions
            for pill in pill_pos:
                points.append((pill.get('x'), -pill.get('y')))

            # Build the top workplane ONCE
            top_plane = (
                result
                .faces('>Z')
                .workplane()
                .center(-pallet_x/2, +pallet_y/2)
            )

            # Unique, sorted X/Y for grid lines
            xs = sorted({x for x, _ in points})
            ys = sorted({y for _, y in points})

            x_lines = []
            y_lines = []
            
            # --- Vertical cuts (thin rectangles at midpoints between Xs) ---
            for i in range(len(xs) - 1):
                x_mid = (xs[i] + xs[i+1]) / 2.0
                x_lines.append(x_mid)
                result_temp = (
                    top_plane
                    .moveTo(x_mid, -short_side/2)
                    .rect(cut_width, short_side, centered=True)
                    .cutBlind(-cut_depth)
                )
                result = result.intersect(result_temp)
            
            # --- Horizontal cuts ---
            for j in range(len(ys) - 1):
                y_mid = (ys[j] + ys[j+1]) / 2.0
                y_lines.append(y_mid)
                result_temp = (
                    top_plane
                    .moveTo(long_side/2, y_mid)
                    .rect(long_side, cut_width, centered=True)
                    .cutBlind(-cut_depth)
                )
                result = result.intersect(result_temp)

            # Small epsilon to avoid zero-size rectangles
            EPS = 1e-6
            
            # --- Ensure midlines are sorted and define blister boundaries ---
            x_mids = sorted(x_lines)
            y_mids = sorted(y_lines)
            X_MIN, X_MAX = 0.0, float(long_side)
            Y_MIN, Y_MAX = -float(short_side), 0.0

            # --- Handle deleted pills ---
            for del_pill in dels_pills:
                del_x = float(del_pill.get('x'))
                del_y = -float(del_pill.get('y'))

                x_left, x_right = span_around(del_x, x_mids, X_MIN, X_MAX)
                y_bottom, y_top = span_around(del_y, y_mids, Y_MIN, Y_MAX)

                rect_w = max(EPS, x_right - x_left)
                rect_h = max(EPS, y_top - y_bottom)

                cx = (x_left + x_right) / 2.0
                cy = (y_bottom + y_top) / 2.0

                result_temp = (
                    top_plane
                    .moveTo(cx, cy)
                    .rect(rect_w, rect_h, centered=True)
                    .cutBlind(-slot_depth)
                )
                result = result.intersect(result_temp)

            # --- NEW: Process features (excavate rectangular areas) ---
            for feature in features:
                feat_x = float(feature.get('x'))
                feat_y = -float(feature.get('y'))  # Negative Y in top_plane system
                feat_width = float(feature.get('width'))
                feat_height = float(feature.get('height'))
                feat_thickness = float(feature.get('thickness'))
                feat_shape = feature.get('shape', 'Rectangle')
                
                # Excavate the feature
                if feat_shape == "Rectangle":
                    result_temp = (
                        top_plane
                        .moveTo(feat_x, feat_y)
                        .rect(feat_width, feat_height, centered=True)
                        .cutBlind(-feat_thickness)
                    )
                    result = result.intersect(result_temp)
                # Add other shapes if needed in the future

            # --- Generate G-code with feature-aware cutting ---
            config = CNCConfig(
                units=Units.METRIC,
                feed_rate=300.0,
                spindle_speed=0,
                plunge_rate=50.0,
                safe_z=5.0,
                work_z=0.0,
                cut_depth=-cut_depth,
                step_down=1.0,
                tool_diameter=cut_width
            )

            gen = GCodeGenerator(config)
            gen.add_header(f"Blister Pack Cutting - {rows}x{columns}")

            # Modified grid cutting with feature awareness
            # You'll need to modify the GCodeGenerator to accept features parameter
            # For now, using the existing method
            gen.cut_grid_lines(
                x_lines=x_lines,
                y_lines=y_lines,
                x_min=0.0,
                x_max=long_side,
                y_min=-short_side,
                y_max=0.0,
                depth=-cut_depth,
                single_pass=True
            )
            
            # Add deeper cuts for features along grid lines
            for x_line in x_lines:
                for feature in features:
                    feat_x = float(feature.get('x'))
                    feat_y = -float(feature.get('y'))
                    feat_width = float(feature.get('width'))
                    feat_height = float(feature.get('height'))
                    feat_thickness = float(feature.get('thickness'))
                    
                    # Check if vertical line intersects this feature
                    feat_left = feat_x - feat_width / 2
                    feat_right = feat_x + feat_width / 2
                    feat_bottom = feat_y - feat_height / 2
                    feat_top = feat_y + feat_height / 2
                    
                    if feat_left <= x_line <= feat_right:
                        # Add extra cut through this feature
                        gen.add_comment(f"Extra cut through feature at x={x_line}")
                        gen.add_line_move(x_line, feat_bottom, config.safe_z)
                        gen.add_line_move(x_line, feat_bottom, -feat_thickness)
                        gen.add_line_move(x_line, feat_top, -feat_thickness)
                        gen.add_line_move(x_line, feat_top, config.safe_z)
            
            for y_line in y_lines:
                for feature in features:
                    feat_x = float(feature.get('x'))
                    feat_y = -float(feature.get('y'))
                    feat_width = float(feature.get('width'))
                    feat_height = float(feature.get('height'))
                    feat_thickness = float(feature.get('thickness'))
                    
                    # Check if horizontal line intersects this feature
                    feat_left = feat_x - feat_width / 2
                    feat_right = feat_x + feat_width / 2
                    feat_bottom = feat_y - feat_height / 2
                    feat_top = feat_y + feat_height / 2
                    
                    if feat_bottom <= y_line <= feat_top:
                        # Add extra cut through this feature
                        gen.add_comment(f"Extra cut through feature at y={y_line}")
                        gen.add_line_move(feat_left, y_line, config.safe_z)
                        gen.add_line_move(feat_left, y_line, -feat_thickness)
                        gen.add_line_move(feat_right, y_line, -feat_thickness)
                        gen.add_line_move(feat_right, y_line, config.safe_z)
            
            gen.add_footer()
            
            # Save G-code
            gcode_path = "output.nc"
            gen.save_to_file(gcode_path)

            # --- Handle pill shapes ---
            if pill_shape == "Oval":
                result_temp = (
                    top_plane
                    .pushPoints(points)
                    .ellipse(pill_height / 2, pill_width / 2)
                    .cutBlind(pill_thickness)
                )
                result = result.intersect(result_temp)
            elif pill_shape == "Circle":
                result_temp = (
                    top_plane
                    .pushPoints(points)
                    .circle(pill_height/2)
                    .cutBlind(pill_thickness)
                )
                result = result.intersect(result_temp)
            elif pill_shape == "Rectangle":
                result_temp = (
                    top_plane
                    .pushPoints(points)
                    .rect(pill_height, pill_width)
                    .cutBlind(pill_thickness)
                )
                result = result.intersect(result_temp)
            elif pill_shape == "Fillet":
                if pill_height > pill_width:
                    fillet_radius = pill_width / 2
                    
                    result_temp = (
                        top_plane
                        .pushPoints(points)
                        .rect(pill_height - pill_width, pill_width)
                        .cutBlind(pill_thickness)
                    )
                    result = result.intersect(result_temp)
                    
                    left_points = [(x - (pill_height - pill_width)/2, y) for x, y in points]
                    result_l = (
                        top_plane
                        .pushPoints(left_points)
                        .circle(fillet_radius)
                        .cutBlind(pill_thickness)
                    )
                    
                    right_points = [(x + (pill_height - pill_width)/2, y) for x, y in points]
                    result_r = (
                        top_plane
                        .pushPoints(right_points)
                        .circle(fillet_radius)
                        .cutBlind(pill_thickness)
                    )
                    result = result.intersect(result_l).intersect(result_r)
                    
                else:
                    fillet_radius = pill_height / 2
                    
                    result_temp = (
                        top_plane
                        .pushPoints(points)
                        .rect(pill_height, pill_width - pill_height)
                        .cutBlind(pill_thickness)
                    )
                    result = result.intersect(result_temp)
                    
                    top_points = [(x, y + (pill_width - pill_height)/2) for x, y in points]
                    result_t = (
                        top_plane
                        .pushPoints(top_points)
                        .circle(fillet_radius)
                        .cutBlind(pill_thickness)
                    )
                    
                    bottom_points = [(x, y - (pill_width - pill_height)/2) for x, y in points]
                    result_b = (
                        top_plane
                        .pushPoints(bottom_points)
                        .circle(fillet_radius)
                        .cutBlind(pill_thickness)
                    )
                    result = result.intersect(result_t).intersect(result_b)
            else:
                return {'error1': 'unknown_pill_shape', 'pill_shape': pill_shape}, 400

        else:
            return {'error2': 'unknown_command', 'command': command}, 400

        # Create temporary file for STEP
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.step')
        temp_path = temp_file.name
        temp_file.close()

        # Export the model as STEP file
        cq.exporters.export(result, temp_path)
        step_file_path = temp_path
        nc_file_path = gcode_path
        
        # Create a ZIP file in memory
        memory_zip = BytesIO()
        with zipfile.ZipFile(memory_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(step_file_path, os.path.basename(step_file_path))
            zf.write(nc_file_path, os.path.basename(nc_file_path))
        
        memory_zip.seek(0)
        
        return send_file(
            memory_zip,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'blister_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )

    except Exception as e:
        print('Error:', str(e))
        return {'error3': str(e)}, 400


@app.route('/', methods=['GET'])
def home():
    """
    Simple endpoint to check if server is running
    """
    return {'message': 'Server is running', 'endpoint': 'POST /convert'}


if __name__ == '__main__':
    print("Server running on http://localhost:8000")
    print("Endpoint: POST /convert")
    app.run(host='0.0.0.0', port=8000, debug=False)