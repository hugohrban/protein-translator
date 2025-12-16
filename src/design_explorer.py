import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

# Set page configuration
st.set_page_config(page_title="Protein Design Explorer", layout="wide")

# Title
st.title("🧬 Protein Design Explorer")

# Load data
@st.cache_data
def load_designs(filepath):
    """Load designs from JSONL file."""
    designs = []
    with open(filepath, 'r') as f:
        for line in f:
            designs.append(json.loads(line.strip()))
    df = pd.DataFrame(designs)
    return df

# Path to designs file
designs_path = Path(__file__).parent / "database" / "designs.jsonl"

try:
    df = load_designs(designs_path)
    
    # Sidebar for axis selection
    st.sidebar.header("Plot Configuration")
    
    # Available numeric columns (excluding sequence and path columns)
    numeric_cols = ['plddt', 'rmsd', 'tm_score', 'beam_size', 'plddt_scaling_factor', 
                    'distance_threshold', 'clustering_proportion']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    x_axis = st.sidebar.selectbox(
        "X-axis",
        options=available_cols,
        index=available_cols.index('plddt') if 'plddt' in available_cols else 0
    )
    
    y_axis = st.sidebar.selectbox(
        "Y-axis",
        options=available_cols,
        index=available_cols.index('rmsd') if 'rmsd' in available_cols else 0
    )
    
    # Optional color dimension
    color_by = st.sidebar.selectbox(
        "Color by (optional)",
        options=['None'] + available_cols,
        index=0
    )
    
    # Axis limits
    st.sidebar.header("Axis Limits (Optional)")
    
    # X-axis limits
    x_min_default, x_max_default = float(df[x_axis].min()), float(df[x_axis].max())
    # Set default upper bound to 5 for RMSD
    if x_axis == 'rmsd' and x_max_default > 5:
        x_max_display = 5.0
        use_x_limits_default = True
    else:
        x_max_display = x_max_default
        use_x_limits_default = False
    
    use_x_limits = st.sidebar.checkbox("Limit X-axis range", value=use_x_limits_default)
    if use_x_limits:
        x_min = st.sidebar.number_input(f"Min {x_axis}", value=x_min_default, 
                                       min_value=x_min_default, max_value=x_max_default)
        x_max = st.sidebar.number_input(f"Max {x_axis}", value=x_max_display,
                                       min_value=x_min_default, max_value=x_max_default)
    else:
        x_min, x_max = x_min_default, x_max_default
    
    # Y-axis limits
    y_min_default, y_max_default = float(df[y_axis].min()), float(df[y_axis].max())
    # Set default upper bound to 5 for RMSD
    if y_axis == 'rmsd' and y_max_default > 5:
        y_max_display = 5.0
        use_y_limits_default = True
    else:
        y_max_display = y_max_default
        use_y_limits_default = False
    
    use_y_limits = st.sidebar.checkbox("Limit Y-axis range", value=use_y_limits_default)
    if use_y_limits:
        y_min = st.sidebar.number_input(f"Min {y_axis}", value=y_min_default,
                                       min_value=y_min_default, max_value=y_max_default)
        y_max = st.sidebar.number_input(f"Max {y_axis}", value=y_max_display,
                                       min_value=y_min_default, max_value=y_max_default)
    else:
        y_min, y_max = y_min_default, y_max_default
    
    # Filter dataframe based on limits
    df_filtered = df[(df[x_axis] >= x_min) & (df[x_axis] <= x_max) & 
                     (df[y_axis] >= y_min) & (df[y_axis] <= y_max)]
    
    # Display statistics
    st.sidebar.header("Dataset Statistics")
    st.sidebar.metric("Total Designs", len(df))
    st.sidebar.metric("Filtered Designs", len(df_filtered))
    st.sidebar.metric(f"Mean {x_axis}", f"{df_filtered[x_axis].mean():.4f}")
    st.sidebar.metric(f"Mean {y_axis}", f"{df_filtered[y_axis].mean():.4f}")
    
    # Prepare hover data (all hyperparameters except sequence and reference_structure)
    hover_cols = [col for col in df.columns 
                  if col not in ['sequence', 'reference_structure']]
    
    # Create the scatter plot
    fig = px.scatter(
        df_filtered,
        x=x_axis,
        y=y_axis,
        color=color_by if color_by != 'None' else None,
        hover_data=hover_cols,
        title=f"{y_axis.upper()} vs {x_axis.upper()}",
        labels={
            x_axis: x_axis.upper(),
            y_axis: y_axis.upper()
        },
        height=600
    )
    
    # Update layout for better interactivity
    fig.update_layout(
        dragmode='select',
        hovermode='closest',
        clickmode='event+select'
    )
    
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    
    # Display selection info
    st.subheader("Selection Information")
    st.info("💡 Use the box select or lasso select tool in the plot to select multiple points. Click on individual points to see details.")
    
    # Show selected design IDs with 60% width
    col_plot, col_spacer = st.columns([0.6, 0.4])
    with col_plot:
        selection = st.plotly_chart(fig, use_container_width=True, key='plot', on_select='rerun')
    
    if selection and 'selection' in selection and 'points' in selection['selection']:
        selected_indices = [point['point_index'] for point in selection['selection']['points']]
        
        if selected_indices:
            st.subheader(f"Selected Designs ({len(selected_indices)})")
            
            # Display selected design IDs
            selected_df = df_filtered.iloc[selected_indices][['design_id', "rmsd", "plddt", "tm_score"]]
            st.dataframe(selected_df, use_container_width=True)
            
            # Button to copy IDs
            ids_string = ' '.join(df_filtered.iloc[selected_indices]['design_id'].tolist())
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text_area("Selected Design IDs", ids_string, height=100, key='ids_text')
            with col2:
                st.write("")
                st.write("")
                if st.button("📋 Copy IDs", use_container_width=True):
                    st.code(ids_string, language=None)
                    st.success("IDs displayed above - copy from the code block!")
            
            # Option to view full details
            with st.expander("View Full Details of Selected Designs"):
                for idx in selected_indices:
                    design = df_filtered.iloc[idx]
                    design_id = design['design_id']
                    st.markdown(f"**Design ID:** `{design_id}`")
                    
                    # Display hyperparameters in columns
                    cols = st.columns(3)
                    params = {k: v for k, v in design.items() 
                             if k not in ['design_id', 'sequence', 'reference_structure']}
                    
                    for i, (key, value) in enumerate(params.items()):
                        col_idx = i % 3
                        cols[col_idx].metric(key, f"{value}")
                    
                    # Button to show structure in PyMOL
                    structure_path = Path(__file__).parent / "database" / "designs" / f"{design_id}.pdb"
                    abs_structure_path = structure_path.resolve()
                    
                    if structure_path.exists():
                        col, _ = st.columns([1, 1])
                        with col:
                            if st.button(f"🔬 Open in PyMOL", key=f"pymol_{design_id}"):
                                import subprocess
                                try:
                                    subprocess.Popen(['pymol', str(abs_structure_path)])
                                    st.success("Opening in PyMOL...")
                                except FileNotFoundError:
                                    st.error("PyMOL not found. Please install PyMOL or use the command above.")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    else:
                        st.caption(f"⚠️ Structure file not found: {structure_path.name}")
                    
                    st.divider()
    
    # Summary table
    st.subheader("All Designs Summary")
    display_cols = ['design_id', 'plddt', 'rmsd', 'tm_score', 'beam_size', 
                    'plddt_scaling_factor', 'distance_threshold']
    display_cols = [col for col in display_cols if col in df_filtered.columns]
    st.dataframe(df_filtered[display_cols], use_container_width=True, height=300)
    
except FileNotFoundError:
    st.error(f"❌ Could not find designs file at: {designs_path}")
    st.info("Please ensure the designs.jsonl file exists in src/database/")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.exception(e)
