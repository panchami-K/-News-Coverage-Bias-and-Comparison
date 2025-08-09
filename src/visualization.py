import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Dict, Any

class SpectrumAnalysisVisualizer:
    """
    A reusable class for creating interactive visualizations of political spectrum data.
    
    Assumes data has columns: 'topic_id', 'predicted_leaning', 'headline', 'perspective_summary'
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualizer with a DataFrame.
        
        Args:
            df: DataFrame containing columns ['topic_id', 'predicted_leaning', 'headline', 'perspective_summary']
        """
        self.df = df
        self.required_columns = ['topic_id', 'predicted_leaning', 'headline', 'perspective_summary']
        self._validate_data()
        
    def _validate_data(self):
        """Validate that the DataFrame has required columns."""
        missing_cols = [col for col in self.required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def plot_topic_distribution(self, color_sequence: Optional[List[str]] = None) -> go.Figure:
        """
        Create an interactive bar chart showing articles per topic.
        
        Args:
            color_sequence: Custom color sequence for the bars
            
        Returns:
            Plotly figure object
        """
        if color_sequence is None:
            color_sequence = px.colors.qualitative.Set2
            
        topic_counts = self.df['topic_id'].value_counts().sort_index()
        
        fig = px.bar(
            x=topic_counts.index, 
            y=topic_counts.values,
            color=topic_counts.index,
            title='Number of Articles per Topic Cluster',
            labels={'x': 'Topic ID', 'y': 'Number of Articles'},
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(showlegend=False)
        return fig
    
    def plot_political_leaning_distribution(self, color_sequence: Optional[List[str]] = None) -> go.Figure:
        """
        Create an interactive bar chart showing distribution of political leanings.
        
        Args:
            color_sequence: Custom color sequence for the bars
            
        Returns:
            Plotly figure object
        """
        if color_sequence is None:
            color_sequence = px.colors.qualitative.Pastel
            
        leaning_counts = self.df['predicted_leaning'].value_counts()
        
        fig = px.bar(
            x=leaning_counts.index,
            y=leaning_counts.values,
            color=leaning_counts.index,
            title='Distribution of Predicted Political Leanings',
            labels={'x': 'Political Leaning', 'y': 'Number of Articles'},
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(showlegend=False)
        return fig
    
    def plot_topic_vs_leaning_heatmap(self, color_scale: str = 'Viridis') -> go.Figure:
        """
        Create an interactive heatmap showing topic vs political leaning distribution.
        
        Args:
            color_scale: Color scale for the heatmap
            
        Returns:
            Plotly figure object
        """
        heatmap_data = pd.crosstab(self.df['topic_id'], self.df['predicted_leaning'])
        
        fig = px.imshow(
            heatmap_data,
            text_auto=True,
            color_continuous_scale=color_scale,
            aspect='auto',
            labels=dict(x="Political Leaning", y="Topic ID", color="Article Count"),
            title="Number of Articles by Topic and Political Leaning"
        )
        fig.update_layout(xaxis_title="Political Leaning", yaxis_title="Topic ID")
        return fig
    
    def plot_topic_perspective_pie(self, topic_id: int, color_sequence: Optional[List[str]] = None) -> go.Figure:
        """
        Create a pie chart showing perspective share for a specific topic.
        
        Args:
            topic_id: The topic ID to analyze
            color_sequence: Custom color sequence for the pie chart
            
        Returns:
            Plotly figure object
        """
        if color_sequence is None:
            color_sequence = px.colors.qualitative.Prism
            
        topic_df = self.df[self.df['topic_id'] == topic_id]
        
        if topic_df.empty:
            raise ValueError(f"No articles found for topic {topic_id}")
        
        fig = px.pie(
            topic_df, 
            names='predicted_leaning', 
            title=f'Perspective Share for Topic {topic_id}',
            color_discrete_sequence=color_sequence
        )
        fig.update_traces(textinfo='percent+label')
        return fig
    
    def create_spectrum_table(self, topic_id: int, max_articles_per_leaning: int = 2) -> go.Figure:
        """
        Create an interactive table showing side-by-side perspectives for a topic.
        
        Args:
            topic_id: The topic ID to analyze
            max_articles_per_leaning: Maximum number of articles to show per political leaning
            
        Returns:
            Plotly figure object
        """
        leanings_order = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
        topic_df = self.df[self.df['topic_id'] == topic_id]
        
        if topic_df.empty:
            raise ValueError(f"No articles found for topic {topic_id}")
        
        spectrum_data = []
        for leaning in leanings_order:
            articles = topic_df[topic_df['predicted_leaning'] == leaning]
            for idx, row in articles.head(max_articles_per_leaning).iterrows():
                spectrum_data.append({
                    "Political Leaning": leaning,
                    "Headline": row['headline'],
                    "Summary": row['perspective_summary']
                })
        
        spectrum_df = pd.DataFrame(spectrum_data)
        
        if spectrum_df.empty:
            raise ValueError(f"No articles with recognized political leanings found for topic {topic_id}")
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(spectrum_df.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[spectrum_df[col] for col in spectrum_df.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=10, color='black'),
                height=30
            )
        )])
        
        fig.update_layout(
            title=f"Spectrum Tab: Side-by-Side Perspectives for Topic {topic_id}",
            height=600
        )
        return fig
    
    def plot_static_heatmap(self, figsize: tuple = (10, 6), cmap: str = 'YlGnBu') -> None:
        """
        Create a static annotated heatmap using seaborn.
        
        Args:
            figsize: Figure size tuple (width, height)
            cmap: Colormap for the heatmap
        """
        heatmap_data = pd.crosstab(self.df['topic_id'], self.df['predicted_leaning'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap=cmap)
        plt.title('Number of Articles by Topic and Political Leaning')
        plt.xlabel('Political Leaning')
        plt.ylabel('Topic ID')
        plt.tight_layout()
        plt.show()
    
    def generate_all_visualizations(self, topic_id: int = 0, show_static: bool = True) -> Dict[str, Any]:
        """
        Generate all visualizations and return them as a dictionary.
        
        Args:
            topic_id: Topic ID for topic-specific visualizations
            show_static: Whether to show the static matplotlib plot
            
        Returns:
            Dictionary containing all generated figures
        """
        figures = {}
        
        print("--- Interactive Topic Distribution ---")
        figures['topic_distribution'] = self.plot_topic_distribution()
        figures['topic_distribution'].show()
        
        print("\n--- Interactive Political Leaning Distribution ---")
        figures['leaning_distribution'] = self.plot_political_leaning_distribution()
        figures['leaning_distribution'].show()
        
        print("\n--- Interactive Heatmap: Topic vs. Political Leaning ---")
        figures['heatmap'] = self.plot_topic_vs_leaning_heatmap()
        figures['heatmap'].show()
        
        try:
            print(f"\n--- Interactive Pie Chart: Perspective Share for Topic {topic_id} ---")
            figures['pie_chart'] = self.plot_topic_perspective_pie(topic_id)
            figures['pie_chart'].show()
            
            print(f"\n--- Interactive Spectrum Tab: Topic {topic_id} ---")
            figures['spectrum_table'] = self.create_spectrum_table(topic_id)
            figures['spectrum_table'].show()
        except ValueError as e:
            print(f"Warning: {e}")
        
        if show_static:
            print("\n--- Static Annotated Heatmap (Seaborn) ---")
            self.plot_static_heatmap()
        
        print("\nAll visualizations complete!")
        return figures
    
    def get_topic_summary(self, topic_id: int) -> Dict[str, Any]:
        """
        Get a summary of articles and perspectives for a specific topic.
        
        Args:
            topic_id: The topic ID to summarize
            
        Returns:
            Dictionary containing topic summary statistics
        """
        topic_df = self.df[self.df['topic_id'] == topic_id]
        
        if topic_df.empty:
            return {"error": f"No articles found for topic {topic_id}"}
        
        summary = {
            "topic_id": topic_id,
            "total_articles": len(topic_df),
            "unique_leanings": topic_df['predicted_leaning'].nunique(),
            "leaning_distribution": topic_df['predicted_leaning'].value_counts().to_dict(),
            "sample_headlines": topic_df['headline'].head(3).tolist()
        }
        
        return summary


# Example usage:
if __name__ == "__main__":
    # Assuming you have a DataFrame 'df' with the required columns
    # df = pd.read_csv('your_spectrum_data.csv')
    
    # Create visualizer instance
    # visualizer = SpectrumAnalysisVisualizer(df)
    
    # Generate all visualizations
    # figures = visualizer.generate_all_visualizations(topic_id=0)
    
    # Or create individual visualizations
    # fig1 = visualizer.plot_topic_distribution()
    # fig1.show()
    
    # Get topic summary
    # summary = visualizer.get_topic_summary(topic_id=0)
    # print(summary)
    
    pass