from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import uuid
from .mouse import (
    initialize_grid_with_cheese_types, print_grid_with_cheese_types, 
    move, get_reward, ACTIONS
)
from .models import GameState, Trajectory, HumanFeedback, PolicyModel, TrainingSession

def project5_landing(request):
    """Landing page for Project 5 with environment test"""
    
    # Test the mouse environment using your uploaded file
    try:
        grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()
        
        # Convert grid to symbols for display
        symbols = {
            0: '.',  # EMPTY
            1: 'M',  # MOUSE
            2: 'C',  # CHEESE
            3: 'T',  # TRAP
            4: '#',  # WALL
            5: 'O'   # ORGANIC_CHEESE
        }
        
        grid_display = []
        for row in grid:
            grid_display.append([symbols[cell] for cell in row])
        
        # Test a move
        test_grid = grid.copy()
        test_grid = move('right', test_grid)
        new_mouse_pos = tuple(np.argwhere(test_grid == 1)[0]) if len(np.argwhere(test_grid == 1)) > 0 else mouse_pos
        reward = get_reward(new_mouse_pos, test_grid)
        
        # Get database statistics
        trajectory_count = Trajectory.objects.count()
        feedback_count = HumanFeedback.objects.count()
        policy_count = PolicyModel.objects.count()
        session_count = TrainingSession.objects.count()
        
        html_content = f"""
        <h1>Project 5: Reinforcement Learning with Human Feedback</h1>
        <h2>Mouse Environment Test [SUCCESS]</h2>
        
        <p><strong>Using your uploaded mouse.py file!</strong></p>
        
        <h3>Environment &amp; Database Setup Successful!</h3>
        
        <div style="background: #e6ffe6; padding: 15px; border-radius: 5px; border-left: 4px solid green; margin: 20px 0;">
            <h4>Database Status:</h4>
            <ul>
                <li><strong>Trajectories:</strong> {trajectory_count} stored</li>
                <li><strong>Human Feedback:</strong> {feedback_count} entries</li>
                <li><strong>Policy Models:</strong> {policy_count} saved</li>
                <li><strong>Training Sessions:</strong> {session_count} recorded</li>
            </ul>
        </div>
        
        <h4>Legend:</h4>
        <ul>
            <li><strong>M</strong> = Mouse (player)</li>
            <li><strong>C</strong> = Regular Cheese (+10 reward)</li>
            <li><strong>O</strong> = Organic Cheese (+10 reward, but we want to avoid it via RLHF)</li>
            <li><strong>T</strong> = Trap (-50 reward)</li>
            <li><strong>#</strong> = Wall (impassable)</li>
            <li><strong>.</strong> = Empty space (-0.2 reward per step)</li>
        </ul>
        
        <h4>Random Generated Grid:</h4>
        <pre style="font-family: monospace; font-size: 16px; background: #f0f0f0; padding: 15px; border-radius: 5px; line-height: 1.5;">
        """
        
        for row in grid_display:
            html_content += ' '.join(row) + '\n'
        
        html_content += f"""
        </pre>
        
        <h4>Environment Details:</h4>
        <ul>
            <li><strong>Mouse Position:</strong> {mouse_pos}</li>
            <li><strong>Cheese Position:</strong> {cheese_pos}</li>
            <li><strong>Organic Cheese Positions:</strong> {organic_cheese_positions}</li>
            <li><strong>Test Move (right):</strong> New position {new_mouse_pos}, Reward: {reward}</li>
        </ul>
        
        <h4>Available Actions:</h4>
        <ul>
        """
        
        for action in ACTIONS:
            html_content += f"<li><strong>{action}</strong></li>"
        
        html_content += f"""
        </ul>
        
        <div style="background: #fff8dc; padding: 15px; border-radius: 5px; border-left: 4px solid orange; margin: 20px 0;">
            <h4>Next Steps:</h4>
            <ul>
                <li>[DONE] Database models set up and working</li>
                <li>[TODO] Implement PolicyNetwork (CNN for processing grid states)</li>
                <li>[TODO] Implement REINFORCE training algorithm</li>
                <li>[TODO] Create human feedback collection interface</li>
                <li>[TODO] Implement RLHF with Bradley-Terry preference model</li>
            </ul>
        </div>
        
        <hr>
        <a href="/home/" style="padding: 10px 20px; background-color: #4f7ecb; color: white; text-decoration: none; border-radius: 4px;">Back to Home</a>
        <a href="/project5/test/" style="padding: 10px 20px; background-color: #6b8e23; color: white; text-decoration: none; border-radius: 4px; margin-left: 10px;">Run Direct Test</a>
        <a href="/project5/test-models/" style="padding: 10px 20px; background-color: #cd853f; color: white; text-decoration: none; border-radius: 4px; margin-left: 10px;">Test Database</a>
        """
        
        return HttpResponse(html_content)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return HttpResponse(f"""
        <h1>Project 5: Environment Test</h1>
        <h2 style="color: red;">ERROR in Mouse Environment</h2>
        <p><strong>Error:</strong> {str(e)}</p>
        <p><strong>Error Type:</strong> {type(e).__name__}</p>
        <details>
            <summary>Click for full error details</summary>
            <pre style='background: #ffe6e6; padding: 10px; border-radius: 5px;'>{error_details}</pre>
        </details>
        <p><strong>Troubleshooting:</strong></p>
        <ul>
            <li>Make sure numpy is installed: <code>pip install numpy</code></li>
            <li>Check that mouse.py is in the project5 directory</li>
            <li>Run migrations: <code>python manage.py migrate</code></li>
            <li>Restart the Django server</li>
        </ul>
        <a href="/home/">Back to Home</a>
        """)


def test_environment_direct(request):
    """Direct test of the environment (for debugging)"""
    try:
        import io
        import sys
        
        # Test the functions directly
        grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        print("=== Direct Test of Your Mouse Environment ===")
        print("Initial grid:")
        print_grid_with_cheese_types(grid)
        print(f"Mouse position: {mouse_pos}")
        print(f"Cheese position: {cheese_pos}")
        print(f"Organic cheese positions: {organic_cheese_positions}")
        
        # Test movement
        print("\nTesting movement 'right':")
        test_grid = grid.copy()
        test_grid = move('right', test_grid)
        print_grid_with_cheese_types(test_grid)
        
        # Test reward
        new_mouse_pos = tuple(np.argwhere(test_grid == 1)[0])
        reward = get_reward(new_mouse_pos, test_grid)
        print(f"New mouse position: {new_mouse_pos}")
        print(f"Reward: {reward}")
        
        print("\n=== Test Complete - Everything Working! ===")
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        return HttpResponse(f"<pre style='font-family: monospace; background: #f5f5f5; padding: 20px; border-radius: 5px;'>{output}</pre><p><a href='/project5/'>Back to Project 5</a></p>")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return HttpResponse(f"""
        <h2>Error in Direct Test</h2>
        <p><strong>Error:</strong> {str(e)}</p>
        <pre style='background: #ffe6e6; padding: 10px; border-radius: 5px;'>{error_details}</pre>
        <p><a href='/project5/'>Back to Project 5</a></p>
        """)


def test_models(request):
    """Test database models by creating and retrieving sample data"""
    try:
        # Create a test trajectory
        trajectory_id = str(uuid.uuid4())
        
        # Create sample trajectory
        trajectory = Trajectory.objects.create(
            trajectory_id=trajectory_id,
            total_reward=15.6,
            total_steps=8,
            cheese_collected=1,
            organic_cheese_collected=0,
            traps_hit=0,
            episode_ended=True,
            end_reason="cheese",
            policy_version="test_v1"
        )
        
        # Create sample game states
        grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()
        
        for step in range(3):
            game_state = GameState.objects.create(
                trajectory_id=trajectory_id,
                mouse_position=str(mouse_pos),
                reward=-0.2,
                step_number=step,
                action_taken=['up', 'right', 'down'][step]
            )
            game_state.set_grid(grid)
            game_state.save()
        
        # Create sample feedback
        feedback = HumanFeedback.objects.create(
            trajectory_a_id=trajectory_id,
            trajectory_b_id="test_trajectory_b",
            preferred_trajectory="A",
            feedback_reason="Trajectory A avoided the organic cheese better",
            confidence_level=4
        )
        
        # Get statistics
        trajectory_count = Trajectory.objects.count()
        game_state_count = GameState.objects.count()
        feedback_count = HumanFeedback.objects.count()
        
        # Get the created objects
        recent_trajectories = Trajectory.objects.all()[:5]
        recent_feedback = HumanFeedback.objects.all()[:5]
        
        html_content = f"""
        <h1>Database Models Test</h1>
        <h2>[SUCCESS] Database Models Working Successfully!</h2>
        
        <h3>Test Results:</h3>
        <ul>
            <li><strong>Created test trajectory:</strong> {trajectory_id[:8]}...</li>
            <li><strong>Created 3 game states</strong></li>
            <li><strong>Created sample feedback</strong></li>
        </ul>
        
        <h3>Database Statistics:</h3>
        <ul>
            <li><strong>Total Trajectories:</strong> {trajectory_count}</li>
            <li><strong>Total Game States:</strong> {game_state_count}</li>
            <li><strong>Total Feedback Entries:</strong> {feedback_count}</li>
        </ul>
        
        <h3>Recent Trajectories:</h3>
        <ul>
        """
        
        for traj in recent_trajectories:
            html_content += f"<li>{traj.trajectory_id[:8]}... - Reward: {traj.total_reward}, Steps: {traj.total_steps}, Reason: {traj.end_reason}</li>"
        
        html_content += """
        </ul>
        
        <h3>Recent Feedback:</h3>
        <ul>
        """
        
        for fb in recent_feedback:
            html_content += f"<li>Preferred {fb.preferred_trajectory} (Confidence: {fb.confidence_level}/5) - {fb.feedback_reason[:50]}...</li>"
        
        html_content += f"""
        </ul>
        
        <div style="background: #e6ffe6; padding: 15px; border-radius: 5px; border-left: 4px solid green; margin: 20px 0;">
            <h4>[SUCCESS] All Database Models Working!</h4>
            <p>The database is ready to store trajectories, game states, human feedback, and policy models.</p>
        </div>
        
        <hr>
        <a href="/project5/" style="padding: 10px 20px; background-color: #4f7ecb; color: white; text-decoration: none; border-radius: 4px;">Back to Project 5</a>
        <a href="/admin/" style="padding: 10px 20px; background-color: #6b8e23; color: white; text-decoration: none; border-radius: 4px; margin-left: 10px;">View in Admin</a>
        """
        
        return HttpResponse(html_content)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return HttpResponse(f"""
        <h2>ERROR: Database Models Test Failed</h2>
        <p><strong>Error:</strong> {str(e)}</p>
        <pre style='background: #ffe6e6; padding: 10px; border-radius: 5px;'>{error_details}</pre>
        <p><strong>Try running:</strong></p>
        <ul>
            <li><code>python manage.py makemigrations project5</code></li>
            <li><code>python manage.py migrate</code></li>
        </ul>
        <p><a href='/project5/'>Back to Project 5</a></p>
        """)