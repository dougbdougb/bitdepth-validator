import click
import json
from rich.console import Console
from rich.table import Table
from .core import load_image, analyze_bit_depth, detect_histogram_combing
from .stress import run_stress_test
from .noise import run_noise_analysis

console = Console()

@click.group()
def main():
    """16-Bit Image Validator CLI"""
    pass

@main.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--intensity', default=15.0, help='Severity of the S-Curve (10-20 recommended)')
def stress(filepath, intensity):
    """Run an Adjustment Headroom (Stress) Test."""
    
    console.print(f"[bold yellow]Running Stress Test[/bold yellow] on {filepath}...")
    console.print(f"Applying S-Curve with intensity: {intensity}")
    
    try:
        img = load_image(filepath)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    # Run the test
    result = run_stress_test(img, intensity)
    
    # Display Results
    table = Table(title="Headroom / Stress Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Banding Score", f"{result['banding_metric']}%")
    
    status_color = "green" if result['passed'] else "bold red"
    status_text = "PASSED (Robust)" if result['passed'] else "FAILED (Banding Detected)"
    
    table.add_row("Status", f"[{status_color}]{status_text}[/{status_color}]")
    
    console.print(table)
    
    if not result['passed']:
        console.print("\n[yellow]Warning:[/yellow] This image degraded significantly under contrast adjustment.")

@main.command()
@click.argument('filepath', type=click.Path(exists=True))
def noise(filepath):
    """Run Noise Characterization & Dithering Test."""
    
    console.print(f"[bold yellow]Analyzing Noise Profile:[/bold yellow] {filepath}...")
    
    try:
        img = load_image(filepath)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    # Run the test
    results = run_noise_analysis(img)
    
    # Display Results
    table = Table(title="Noise & Dithering Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("FFT Spike Ratio", str(results.get('fft_spike_ratio', 'N/A')))
    table.add_row("Periodic Patterns?", 
                  f"[bold red]Detected[/bold red]" if results.get('has_periodic_patterns') else "[green]None Detected[/green]")
    
    if 'avg_channel_correlation' in results:
        table.add_row("Channel Correlation", str(results['avg_channel_correlation']))
        interp_color = "green" if results['interpretation'] == "Natural" else "bold yellow"
        table.add_row("Interpretation", f"[{interp_color}]{results['interpretation']}[/{interp_color}]")
    
    console.print(table)

@main.command()
@click.argument('filepath', type=click.Path(exists=True))
def inspect(filepath):
    """Run core quality checks on a single PNG file."""
    
    console.print(f"[bold blue]Analyzing:[/bold blue] {filepath}...")
    
    try:
        img = load_image(filepath)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    # Run Tests
    bit_depth_results = analyze_bit_depth(img)
    comb_results = detect_histogram_combing(img)

    # Display Bit Depth Results
    table = Table(title="Bit Depth Authenticity")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Container Depth", "16-bit")
    table.add_row("Effective Depth", f"{bit_depth_results['effective_depth']}-bit")
    table.add_row("Lowest Active Bit", str(bit_depth_results['lowest_active_bit']))
    table.add_row("Is Padded?", str(bit_depth_results['is_padded']))
    
    console.print(table)
    console.print("")

    # Display Histogram/Upscaling Results
    table2 = Table(title="Upscaling Detection")
    table2.add_column("Metric", style="cyan")
    table2.add_column("Value", style="magenta")
    
    table2.add_row("Comb Strength (FFT)", str(comb_results['comb_strength_score']))
    table2.add_row("Likely Upscaled 8-bit?", 
                   f"[bold red]{comb_results['likely_upscaled_8bit']}[/bold red]" 
                   if comb_results['likely_upscaled_8bit'] 
                   else "[green]No[/green]")
    
    console.print(table2)

if __name__ == '__main__':
    main()
