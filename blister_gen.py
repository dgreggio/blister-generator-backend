import os
from flask import Flask, request, send_file
from flask_cors import CORS
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cadquery as cq
from cadquery.vis import show
import tempfile

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
        },
    }
    """
    """
    netstat -ano | findstr :8000
    taskkill /IM python.exe /F
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        # print('Received JSON:', data)
        
        
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
        """for pill in pill_pos:
            return {"error": f"pill x {pill.get('x')}, pill y {pill.get('y')}"}"""

        """long_side = 98
        short_side = 67
        blister_thickness = 0.5
        rows = 2
        columns = 3
        high_border = 7.5
        low_border = 3.5
        left_border = 7.5
        right_border = 7.5
        pill_shape = 'oval'
        pill_height = 22.5
        pill_width = 12
        pill_thickness = 7"""
        #pill_pos = [{'x': 18.75, 'y': 13.5}, {'x': 36.5, 'y': 13.5}, {'x': 54.25, 'y': 13.5}, {'x': 72.0, 'y': 13.5}, {'x': 89.75, 'y': 13.5}, {'x': 18.75, 'y': 47.0}, {'x': 36.5, 'y': 47.0}, {'x': 54.25, 'y': 47.0}, {'x': 72.0, 'y': 47.0}, {'x': 89.75, 'y': 47.0}]

        #return {"error9": f"received, long side: {long_side}, short side: {short_side}, blister_thickness: {blister_thickness}, rows: {rows}, columns: {columns}, high_border: {high_border}, low_border: {low_border}, left_border: {left_border}, right_border: {right_border}, pill_shape: {pill_shape}, pill_height: {pill_height}, pill_width: {pill_width}, pill_thickness: {pill_thickness}"}, 200
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

            #TO MODIFY
            blister = (cq.Workplane('XY').rect(long_side, short_side).extrude(pallet_height-blister_thickness).translate((long_side/2-pallet_x/2,-short_side/2+pallet_y/2,1)))

            #result = piano + guide_vert1 +  guide_vert2 + guide_orr_1 + guide_orr_2 + guide_orr_3 + blister
            result = piano.union(guide_vert1).union(guide_vert2).union(guide_orr_1).union(guide_orr_2).union(guide_orr_3).union(blister)
            points = []

            """for i in range(rows):
                for j in range(columns):
                    x = -long_side/2 + left_border + pill_width/2 + i*(long_side-left_border-right_border - pill_width)/(rows-1)
                    y = -short_side/2 + high_border + pill_height/2 + j*(short_side-high_border-low_border - pill_height)/(columns-1)
                    points.append((x,y))"""
            """
            pillSpacingX = 0
            pillSpacingY = 0
            if columns > 1:
                pillSpacingX = (long_side - left_border - right_border - (columns * pill_height)) / (columns - 1)
            if rows > 1:
                pillSpacingY = (short_side - high_border - low_border - (rows * pill_width)) / (rows - 1)
            for i in range(rows):
                for j in range(columns):
                    x = left_border + pill_height/2 +  j * (pill_height + pillSpacingX)
                    y = - high_border - pill_width/2 - i * (pill_width + pillSpacingY)
                    points.append((x,y))"""
            
            # Let's find blisters by points
            for pill in pill_pos:
                points.append((pill.get('x'), -pill.get('y')))


            """result = (result.faces('>Z').workplane()
                      .center(- pallet_x/2, + pallet_y/2)
                      .pushPoints(points).ellipse(pill_height/2, pill_width/2)
                      .cutBlind((5))"""
            
            # Build the top workplane ONCE
            top_plane = (
                result
                .faces('>Z')
                .workplane()
                .center(-pallet_x/2, +pallet_y/2)
            )

            # Unique, sorted X/Y
            xs = sorted({x for x, _ in points})
            ys = sorted({y for _, y in points})

            x_lines = []
            y_lines = []            # --- Vertical cuts (thin rectangles at midpoints between Xs) ---
            for i in range(len(xs) - 1):
                x_mid = (xs[i] + xs[i+1]) / 2.0
                x_lines.append(x_mid)
                result_temp = (
                    top_plane
                    .moveTo(x_mid, -short_side/2)         # place at center of strip
                    .rect(cut_width, short_side, centered=True)
                    .cutBlind(-cut_depth)                # NEGATIVE: cut into the solid
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
                    .cutBlind(-cut_depth)                # NEGATIVE
                )
                result = result.intersect(result_temp)

            # Small epsilon to avoid zero-size rectangles if a pill sits exactly on a midline
            EPS = 1e-6
            # --- Ensure midlines are sorted and define blister boundaries ---
            x_mids = sorted(x_lines)          # vertical separators (x = const)
            y_mids = sorted(y_lines)          # horizontal separators (y = const)
            X_MIN, X_MAX = 0.0, float(long_side)
            Y_MIN, Y_MAX = -float(short_side), 0.0

            for del_pill in dels_pills:
                del_x = float(del_pill.get('x'))
                del_y = -float(del_pill.get('y'))  # your code uses negative Y on the top plane

                # Find x and y spans around the deletion point
                x_left, x_right = span_around(del_x, x_mids, X_MIN, X_MAX)
                y_bottom, y_top = span_around(del_y, y_mids, Y_MIN, Y_MAX)

                # Width and height of the deletion rectangle
                rect_w = max(EPS, x_right - x_left)
                rect_h = max(EPS, y_top - y_bottom)

                # Center of the deletion rectangle
                cx = (x_left + x_right) / 2.0
                cy = (y_bottom + y_top) / 2.0

                # Recreate the top workplane on the *current* result
                result_temp = (
                    top_plane
                    .moveTo(cx, cy)
                    .rect(rect_w, rect_h, centered=True)
                    .cutBlind(-slot_depth)
                )
                result = result.intersect(result_temp)
                
            config = CNCConfig(
                units=Units.METRIC,
                feed_rate=300.0,          # Adjust to your blade feed rate
                spindle_speed=0,          # 0 for blade (no spindle)
                plunge_rate=50.0,         # Slower for plunging blade
                safe_z=5.0,               # Safe height above workpiece
                work_z=0.0,               # Top of blister surface
                cut_depth=-cut_depth,     # Use your cut_depth variable
                step_down=1.0,            # Depth per pass (adjust as needed)
                tool_diameter=cut_width   # Use your blade width
            )
            # In your blister_gen.py, after calculating x_lines and y_lines:

            config = CNCConfig(
                units=Units.METRIC,
                feed_rate=300.0,          # Adjust to your blade speed
                spindle_speed=0,          # 0 for blade (no spindle needed)
                plunge_rate=50.0,         # Speed for lowering blade
                safe_z=5.0,               # Safe height above workpiece
                work_z=0.0,               # Top of blister surface
                cut_depth=-cut_depth,     # Negative value for cutting depth
                step_down=1.0,            # Not used for single pass
                tool_diameter=cut_width   # Your blade width
            )

            gen = GCodeGenerator(config)
            gen.add_header(f"Blister Pack Cutting - {rows}x{columns}")

            # Cut the grid with single pass (default)
            gen.cut_grid_lines(
                x_lines=x_lines,
                y_lines=y_lines,
                x_min=0.0,
                x_max=long_side,
                y_min=-short_side,
                y_max=0.0,
                depth=-cut_depth,
                single_pass=True  # Single pass cutting
            )
            gen.add_footer()
            # Save G-code
            gcode_path = "output.nc"
            gen.save_to_file(gcode_path)

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
            elif pill_shape == "Fillet": #problem
                # Calculate fillet radius
                if pill_height > pill_width: # Orrizzontale
                    fillet_radius = pill_width / 2
                    
                    # First cut: rectangle
                    result_temp = (
                        top_plane
                        .pushPoints(points)
                        .rect(pill_height - pill_width, pill_width)
                        .cutBlind(pill_thickness)
                    )
                    result = result.intersect(result_temp)
                    
                    # Second cut: circles at both ends
                    # Left circles
                    left_points = [(x - (pill_height - pill_width)/2, y) for x, y in points]
                    result_l = (
                        top_plane
                        .pushPoints(left_points)
                        .circle(fillet_radius)
                        .cutBlind(pill_thickness)
                    )
                    
                    # Right circles
                    right_points = [(x + (pill_height - pill_width)/2, y) for x, y in points]
                    result_r = (
                        top_plane
                        .pushPoints(right_points)
                        .circle(fillet_radius)
                        .cutBlind(pill_thickness)
                    )
                    result = result.intersect(result_l).intersect(result_r)
                    
                else:
                    fillet_radius = pill_height / 2 #verticale
                    
                    # First cut: rectangle
                    result_temp = (
                        top_plane
                        .pushPoints(points)
                        .rect(pill_height, pill_width - pill_height)
                        .cutBlind(pill_thickness)
                    )
                    result = result.intersect(result_temp)
                    
                    # Second cut: circles at top and bottom
                    # Top circles
                    top_points = [(x, y + (pill_width - pill_height)/2) for x, y in points]
                    result_t = (
                        top_plane
                        .pushPoints(top_points)
                        .circle(fillet_radius)
                        .cutBlind(pill_thickness)
                    )
                    
                    # Bottom circles
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



            
            # You can add more complex pallet logic here using the other parameters
            # For example, creating cavities for pills in a grid pattern
            
        else:
            # Unknown command
            return {'error2': 'unknown_command', 'command': command}, 400

        # Create a temporary file to save the STEP file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.step')
        temp_path = temp_file.name
        temp_file.close()

        # Export the model as STEP file
        cq.exporters.export(result, temp_path)

        # Send the STEP file back to Flutter
        response = send_file(
            temp_path,
            mimetype='application/STEP',
            as_attachment=True,
            download_name='output.step'
        )
        show(result)

        # Clean up: delete temporary file after sending
        @response.call_on_close
        def cleanup():
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return response

    except Exception as e:
        # Return error message if something goes wrong
        print('Error:', str(e))
        return {'error3': str(e)}, 400


@app.route('/', methods=['GET'])
def home():
    """
    Simple endpoint to check if server is running
    """
    return {'message': 'Server is running', 'endpoint': 'POST /convert'}


if __name__ == '__main__':
    # Run the Flask server on localhost port 8000
    print("Server running on http://localhost:8000")
    print("Endpoint: POST /convert")
    app.run(host='0.0.0.0', port=8000, debug=False)


