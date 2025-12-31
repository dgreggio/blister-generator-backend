"""
G-Code Generator for CNC Machining
Supports: rectangles, circles, pockets, drilling patterns, and custom paths
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Units(Enum):
    METRIC = "G21"  # Millimeters
    IMPERIAL = "G20"  # Inches


class FeedMode(Enum):
    PER_MINUTE = "G94"
    PER_REVOLUTION = "G95"


@dataclass
class CNCConfig:
    """Configuration for CNC machine"""
    units: Units = Units.METRIC
    feed_rate: float = 100.0  # mm/min or in/min
    spindle_speed: int = 1000  # RPM
    plunge_rate: float = 50.0  # Feed rate for Z-axis
    safe_z: float = 5.0  # Safe height above workpiece
    work_z: float = 0.0  # Top of workpiece
    cut_depth: float = -2.0  # Depth of cut (negative)
    step_down: float = 0.5  # Depth per pass
    tool_diameter: float = 3.0  # Tool diameter for offsets


class GCodeGenerator:
    """Main G-code generator class"""
    
    def __init__(self, config: CNCConfig = None):
        self.config = config or CNCConfig()
        self.gcode: List[str] = []
        self.current_position = [0.0, 0.0, self.config.safe_z]
        
    def reset(self):
        """Clear generated G-code"""
        self.gcode = []
        self.current_position = [0.0, 0.0, self.config.safe_z]
        
    def add_comment(self, comment: str):
        """Add a comment line"""
        self.gcode.append(f"; {comment}")
        
    def add_header(self, program_name: str = "CNC Program"):
        """Add standard G-code header"""
        self.add_comment("=" * 50)
        self.add_comment(program_name)
        self.add_comment("=" * 50)
        self.gcode.append(f"{self.config.units.value}")  # Set units
        self.gcode.append("G90")  # Absolute positioning
        self.gcode.append(f"{FeedMode.PER_MINUTE.value}")  # Feed rate mode
        self.gcode.append("G17")  # XY plane selection
        self.gcode.append(f"M3 S{self.config.spindle_speed}")  # Start spindle
        self.gcode.append("G4 P2")  # Dwell 2 seconds for spindle to reach speed
        self.add_comment("Moving to safe Z")
        self.gcode.append(f"G0 Z{self.config.safe_z}")
        
    def add_footer(self):
        """Add standard G-code footer"""
        self.add_comment("Program end")
        self.gcode.append(f"G0 Z{self.config.safe_z}")  # Return to safe height
        self.gcode.append("M5")  # Stop spindle
        self.gcode.append("G0 X0 Y0")  # Return to origin
        self.gcode.append("M30")  # Program end
        
    def rapid_move(self, x: float = None, y: float = None, z: float = None):
        """Rapid positioning (G0)"""
        cmd = "G0"
        if x is not None:
            cmd += f" X{x:.4f}"
            self.current_position[0] = x
        if y is not None:
            cmd += f" Y{y:.4f}"
            self.current_position[1] = y
        if z is not None:
            cmd += f" Z{z:.4f}"
            self.current_position[2] = z
        self.gcode.append(cmd)
        
    def linear_move(self, x: float = None, y: float = None, z: float = None, feed: float = None):
        """Linear interpolation move (G1)"""
        cmd = "G1"
        if x is not None:
            cmd += f" X{x:.4f}"
            self.current_position[0] = x
        if y is not None:
            cmd += f" Y{y:.4f}"
            self.current_position[1] = y
        if z is not None:
            cmd += f" Z{z:.4f}"
            self.current_position[2] = z
        if feed is not None:
            cmd += f" F{feed:.1f}"
        self.gcode.append(cmd)
        
    def arc_cw(self, x: float, y: float, i: float, j: float, feed: float = None):
        """Clockwise arc (G2)"""
        cmd = f"G2 X{x:.4f} Y{y:.4f} I{i:.4f} J{j:.4f}"
        if feed:
            cmd += f" F{feed:.1f}"
        self.gcode.append(cmd)
        self.current_position[0] = x
        self.current_position[1] = y
        
    def arc_ccw(self, x: float, y: float, i: float, j: float, feed: float = None):
        """Counter-clockwise arc (G3)"""
        cmd = f"G3 X{x:.4f} Y{y:.4f} I{i:.4f} J{j:.4f}"
        if feed:
            cmd += f" F{feed:.1f}"
        self.gcode.append(cmd)
        self.current_position[0] = x
        self.current_position[1] = y
        
    def rectangle(self, x_start: float, y_start: float, width: float, height: float, 
                  depth: float = None, pocket: bool = False):
        """Cut a rectangle or rectangular pocket"""
        depth = depth or self.config.cut_depth
        self.add_comment(f"Rectangle: {width}x{height} at ({x_start}, {y_start})")
        
        # Calculate passes
        num_passes = math.ceil(abs(depth) / self.config.step_down)
        
        for pass_num in range(num_passes):
            z_depth = self.config.work_z - min((pass_num + 1) * self.config.step_down, abs(depth))
            self.add_comment(f"Pass {pass_num + 1}/{num_passes} at Z={z_depth:.4f}")
            
            # Move to start position
            self.rapid_move(x=x_start, y=y_start)
            self.linear_move(z=z_depth, feed=self.config.plunge_rate)
            
            if pocket:
                # Pocket: spiral from outside to inside
                tool_radius = self.config.tool_diameter / 2
                stepover = self.config.tool_diameter * 0.5  # 50% stepover
                
                current_width = width - 2 * tool_radius
                current_height = height - 2 * tool_radius
                offset = tool_radius
                
                while current_width > 0 and current_height > 0:
                    # Cut rectangular path
                    self.linear_move(x=x_start + offset + current_width, y=y_start + offset, 
                                   feed=self.config.feed_rate)
                    self.linear_move(x=x_start + offset + current_width, y=y_start + offset + current_height)
                    self.linear_move(x=x_start + offset, y=y_start + offset + current_height)
                    self.linear_move(x=x_start + offset, y=y_start + offset)
                    
                    offset += stepover
                    current_width -= 2 * stepover
                    current_height -= 2 * stepover
            else:
                # Outline only
                self.linear_move(x=x_start + width, y=y_start, feed=self.config.feed_rate)
                self.linear_move(x=x_start + width, y=y_start + height)
                self.linear_move(x=x_start, y=y_start + height)
                self.linear_move(x=x_start, y=y_start)
            
            self.rapid_move(z=self.config.safe_z)
            
    def circle(self, center_x: float, center_y: float, radius: float, 
               depth: float = None, pocket: bool = False):
        """Cut a circle or circular pocket"""
        depth = depth or self.config.cut_depth
        self.add_comment(f"Circle: R={radius} at ({center_x}, {center_y})")
        
        num_passes = math.ceil(abs(depth) / self.config.step_down)
        
        for pass_num in range(num_passes):
            z_depth = self.config.work_z - min((pass_num + 1) * self.config.step_down, abs(depth))
            self.add_comment(f"Pass {pass_num + 1}/{num_passes} at Z={z_depth:.4f}")
            
            if pocket:
                # Pocket: spiral from center outward
                tool_radius = self.config.tool_diameter / 2
                stepover = self.config.tool_diameter * 0.5
                current_radius = tool_radius
                
                # Move to center
                self.rapid_move(x=center_x, y=center_y)
                self.linear_move(z=z_depth, feed=self.config.plunge_rate)
                
                while current_radius <= radius - tool_radius:
                    # Move to radius
                    self.linear_move(x=center_x + current_radius, y=center_y, 
                                   feed=self.config.feed_rate)
                    # Cut full circle
                    self.arc_cw(x=center_x + current_radius, y=center_y, 
                               i=-current_radius, j=0, feed=self.config.feed_rate)
                    current_radius += stepover
            else:
                # Outline only
                start_x = center_x + radius
                self.rapid_move(x=start_x, y=center_y)
                self.linear_move(z=z_depth, feed=self.config.plunge_rate)
                self.arc_cw(x=start_x, y=center_y, i=-radius, j=0, 
                           feed=self.config.feed_rate)
            
            self.rapid_move(z=self.config.safe_z)
            
    def drill_pattern(self, positions: List[Tuple[float, float]], 
                     depth: float = None, peck_depth: float = None):
        """Drill holes at specified positions"""
        depth = depth or self.config.cut_depth
        peck_depth = peck_depth or abs(depth)
        
        self.add_comment(f"Drilling {len(positions)} holes")
        self.gcode.append("G81")  # Canned drill cycle
        
        for i, (x, y) in enumerate(positions):
            self.add_comment(f"Hole {i + 1} at ({x:.4f}, {y:.4f})")
            self.rapid_move(x=x, y=y)
            
            if peck_depth < abs(depth):
                # Peck drilling
                current_depth = self.config.work_z
                while current_depth > depth:
                    current_depth = max(current_depth - peck_depth, depth)
                    self.linear_move(z=current_depth, feed=self.config.plunge_rate)
                    self.rapid_move(z=self.config.safe_z)
                    self.rapid_move(z=current_depth + 1)  # Rapid to just above
            else:
                # Simple drilling
                self.linear_move(z=depth, feed=self.config.plunge_rate)
                self.rapid_move(z=self.config.safe_z)
        
        self.gcode.append("G80")  # Cancel canned cycle
        
    def text_engraving(self, text: str, x_start: float, y_start: float, 
                      char_width: float = 5.0, char_height: float = 8.0, depth: float = -0.5):
        """Simple single-line text engraving (very basic)"""
        self.add_comment(f"Engraving text: {text}")
        x_offset = 0
        
        for char in text.upper():
            if char == ' ':
                x_offset += char_width
                continue
                
            # Very simplified character rendering (just vertical lines for demo)
            self.rapid_move(x=x_start + x_offset, y=y_start)
            self.linear_move(z=depth, feed=self.config.plunge_rate)
            self.linear_move(y=y_start + char_height, feed=self.config.feed_rate)
            self.rapid_move(z=self.config.safe_z)
            
            x_offset += char_width
            
    def custom_path(self, points: List[Tuple[float, float]], depth: float = None, closed: bool = True):
        """Follow a custom path defined by points"""
        if len(points) < 2:
            return
            
        depth = depth or self.config.cut_depth
        self.add_comment(f"Custom path with {len(points)} points")
        
        num_passes = math.ceil(abs(depth) / self.config.step_down)
        
        for pass_num in range(num_passes):
            z_depth = self.config.work_z - min((pass_num + 1) * self.config.step_down, abs(depth))
            self.add_comment(f"Pass {pass_num + 1}/{num_passes}")
            
            # Move to first point
            self.rapid_move(x=points[0][0], y=points[0][1])
            self.linear_move(z=z_depth, feed=self.config.plunge_rate)
            
            # Follow path
            for x, y in points[1:]:
                self.linear_move(x=x, y=y, feed=self.config.feed_rate)
            
            # Close path if requested
            if closed:
                self.linear_move(x=points[0][0], y=points[0][1], feed=self.config.feed_rate)
            
            self.rapid_move(z=self.config.safe_z)
    
    def cut_straight_line(self, x_start: float, y_start: float, x_end: float, y_end: float, 
                          depth: float = None, single_pass: bool = False):
        """Cut a straight line from start to end point"""
        depth = depth or self.config.cut_depth
        self.add_comment(f"Straight cut from ({x_start:.2f}, {y_start:.2f}) to ({x_end:.2f}, {y_end:.2f})")
        
        if single_pass:
            # Single pass cutting
            z_depth = self.config.work_z + depth  # depth is negative
            
            # Move to start
            self.rapid_move(x=x_start, y=y_start)
            self.linear_move(z=z_depth, feed=self.config.plunge_rate)
            
            # Cut to end
            self.linear_move(x=x_end, y=y_end, feed=self.config.feed_rate)
            
            # Retract
            self.rapid_move(z=self.config.safe_z)
        else:
            # Multiple pass cutting
            num_passes = math.ceil(abs(depth) / self.config.step_down)
            
            for pass_num in range(num_passes):
                z_depth = self.config.work_z - min((pass_num + 1) * self.config.step_down, abs(depth))
                self.add_comment(f"Pass {pass_num + 1}/{num_passes} at Z={z_depth:.4f}")
                
                # Move to start
                self.rapid_move(x=x_start, y=y_start)
                self.linear_move(z=z_depth, feed=self.config.plunge_rate)
                
                # Cut to end
                self.linear_move(x=x_end, y=y_end, feed=self.config.feed_rate)
                
                # Retract
                self.rapid_move(z=self.config.safe_z)
    
    def cut_grid_lines(self, x_lines: List[float], y_lines: List[float], 
                       x_min: float, x_max: float, y_min: float, y_max: float,
                       depth: float = None, single_pass: bool = True):
        """
        Cut a grid pattern defined by vertical (x_lines) and horizontal (y_lines) cuts.
        
        Args:
            x_lines: List of X coordinates for vertical cuts
            y_lines: List of Y coordinates for horizontal cuts
            x_min, x_max: Bounds for horizontal cuts
            y_min, y_max: Bounds for vertical cuts
            depth: Cutting depth
            single_pass: If True, cut in a single pass. If False, use multiple passes.
        """
        depth = depth or self.config.cut_depth
        
        # Cut all vertical lines (constant X, varying Y)
        if x_lines:
            self.add_comment("="*50)
            self.add_comment(f"Cutting {len(x_lines)} vertical lines")
            self.add_comment("="*50)
            
            for i, x in enumerate(x_lines):
                self.add_comment(f"Vertical cut {i+1}/{len(x_lines)} at X={x:.4f}")
                self.cut_straight_line(x, y_min, x, y_max, depth=depth, single_pass=single_pass)
        
        # Cut all horizontal lines (constant Y, varying X)
        if y_lines:
            self.add_comment("="*50)
            self.add_comment(f"Cutting {len(y_lines)} horizontal lines")
            self.add_comment("="*50)
            
            for i, y in enumerate(y_lines):
                self.add_comment(f"Horizontal cut {i+1}/{len(y_lines)} at Y={y:.4f}")
                self.cut_straight_line(x_min, y, x_max, y, depth=depth, single_pass=single_pass)
    
    def get_gcode(self) -> str:
        """Return the complete G-code as a string"""
        return "\n".join(self.gcode)
    
    def save_to_file(self, filename: str):
        """Save G-code to a file"""
        with open(filename, 'w') as f:
            f.write(self.get_gcode())
        print(f"G-code saved to: {filename}")


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = CNCConfig(
        units=Units.METRIC,
        feed_rate=300.0,
        spindle_speed=12000,
        plunge_rate=100.0,
        safe_z=5.0,
        work_z=0.0,
        cut_depth=-3.0,
        step_down=1.0,
        tool_diameter=6.0
    )
    
    # Create generator
    gen = GCodeGenerator(config)
    
    # Generate program
    gen.add_header("Example CNC Program")
    
    # Cut a rectangle outline
    gen.rectangle(x_start=10, y_start=10, width=50, height=30, pocket=False)
    
    # Cut a circular pocket
    gen.circle(center_x=80, center_y=25, radius=15, pocket=True)
    
    # Drill some holes
    drill_points = [
        (20, 20),
        (40, 20),
        (20, 30),
        (40, 30)
    ]
    gen.drill_pattern(drill_points, depth=-5.0)
    
    # Custom path (triangle)
    triangle = [
        (100, 10),
        (120, 40),
        (80, 40)
    ]
    gen.custom_path(triangle, depth=-2.0, closed=True)
    
    gen.add_footer()
    
    # Output
    print(gen.get_gcode())
    print("\n" + "="*50)
    
    # Save to file
    gen.save_to_file("example_program.nc")
