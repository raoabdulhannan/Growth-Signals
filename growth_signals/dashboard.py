import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import defaultdict
import statistics
from datasets import load_dataset
import base64
import warnings

# Configure the Streamlit page
st.set_page_config(
    page_title="Exploratory Data Analysis: Cohere Wikipedia Dataset",
    page_icon="📊",
    layout="wide"
)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Main title and introduction
st.title("Exploratory Data Analysis: Cohere Wikipedia Dataset")

# Dataset information section
st.markdown("""
## Dataset Overview
### **Cohere Wikipedia Dataset**
* Wikipedia embeddings (35.2M rows)
* Each article divided into paragraphs, each row represents a paragraph
* Embeddings computed using Cohere's [multilingual-22-12](https://cohere.com/blog/multilingual) model
* Data is sorted by number of views an article has

### **Cohere Embedding Model**
* Strong performance
   * Semantic search
   * Customer feedback aggregation
   * Zero-shot content moderation
* Preserves semantic information, results in [clusters](https://storage.googleapis.com/cohere-assets/blog/embeddings/multilingual-embeddings-demo.html?ref=cohere-ai.ghost.io)
* Free API available however trials are limited
""")

# Sidebar configuration for interactive parameters
st.sidebar.header("Analysis Parameters")

# Add interactive sliders for various parameters
sample_size = st.sidebar.slider("Sample Size", 1000, 10000, 5000, 500)
n_clusters = st.sidebar.slider("Number of Clusters (K-means)", 3, 10, 5)
tsne_perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 30)
umap_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 50, 15)
umap_min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 1.0, 0.1)


@st.cache_data
def load_data(sample_size):
    docs = load_dataset(
        "Cohere/wikipedia-22-12-en-embeddings",
        split="train",
        streaming=True
    ).shuffle(seed=42)

    sampled_docs = list(docs.take(sample_size))
    embeddings = np.array([doc['emb'] for doc in sampled_docs])
    titles = [doc['title'] for doc in sampled_docs]
    texts = [doc['text'] for doc in sampled_docs]

    return sampled_docs, embeddings, titles, texts


# Load the dataset
sampled_docs, embeddings, titles, texts = load_data(sample_size)

# Create tabs for organizing different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Document Properties",
    "Dimensionality Reduction",
    "Clustering Analysis",
    "Similarity Analysis"
])


# Document Properties Tab Implementation
with tab1:
    st.header("Document Properties Analysis")

    with st.expander("Understanding Document Properties Analysis"):
        st.markdown("""
        This analysis examines Wikipedia articles at two levels:

        **Article Level Analysis**
        Examines complete Wikipedia articles by combining all their paragraphs:
        - Total article length and structure
        - Number and distribution of paragraphs
        - Vocabulary richness across entire articles
        - Overall semantic characteristics

        **Paragraph Level Analysis**
        Analyzes individual paragraphs within articles:
        - Individual paragraph lengths and properties
        - Embedding characteristics of paragraphs
        - Word usage patterns at paragraph level
        - Relationship between titles and paragraph content
        """)

    @st.cache_data
    def analyze_document_properties(docs):
        try:
            # Initialize dictionaries for article-level aggregation
            article_stats = defaultdict(lambda: {
                'total_length': 0,
                'num_paragraphs': 0,
                'vocab_size': set(),
                'paragraph_lengths': [],
                'embedding_norms': [],
                'sentences': 0,
                'title_length': 0
            })

            # Process each document (paragraph)
            for doc in docs:
                title = doc['title']
                text = doc['text']
                words = text.split()

                # Update article statistics
                article_stats[title]['total_length'] += len(words)
                article_stats[title]['num_paragraphs'] += 1
                article_stats[title]['vocab_size'].update(words)
                article_stats[title]['paragraph_lengths'].append(len(words))
                article_stats[title]['embedding_norms'].append(np.linalg.norm(doc['emb']))
                article_stats[title]['sentences'] += len(text.split('.'))
                article_stats[title]['title_length'] = len(title.split())

            # Create article-level DataFrame
            article_data = []
            for title, stat in article_stats.items():
                try:
                    std_paragraph_length = statistics.stdev(stat['paragraph_lengths']) if len(stat['paragraph_lengths']) > 1 else 0
                except statistics.StatisticsError:
                    std_paragraph_length = 0

                article_data.append({
                    'title': title,
                    'total_words': stat['total_length'],
                    'num_paragraphs': stat['num_paragraphs'],
                    'vocab_size': len(stat['vocab_size']),
                    'avg_paragraph_length': statistics.mean(stat['paragraph_lengths']) if stat['paragraph_lengths'] else 0,
                    'std_paragraph_length': std_paragraph_length,
                    'avg_embedding_norm': statistics.mean(stat['embedding_norms']) if stat['embedding_norms'] else 0,
                    'sentences': stat['sentences'],
                    'title_length': stat['title_length'],
                    'lexical_density': len(stat['vocab_size']) / stat['total_length'] if stat['total_length'] > 0 else 0
                })

            # Create paragraph-level DataFrame
            paragraph_data = [{
                'title': doc['title'],
                'paragraph_length': len(doc['text'].split()),
                'embedding_norm': np.linalg.norm(doc['emb']),
                'sentences': len(doc['text'].split('.')),
                'title_length': len(doc['title'].split()),
                'avg_word_length': len(doc['text']) / len(doc['text'].split()) if doc['text'].split() else 0
            } for doc in docs]

            return pd.DataFrame(article_data), pd.DataFrame(paragraph_data)

        except Exception as e:
            st.error(f"Error in analyze_document_properties: {str(e)}")
            empty_article_df = pd.DataFrame(columns=[
                'title', 'total_words', 'num_paragraphs', 'vocab_size',
                'avg_paragraph_length', 'std_paragraph_length',
                'avg_embedding_norm', 'sentences', 'title_length',
                'lexical_density'
            ])
            empty_paragraph_df = pd.DataFrame(columns=[
                'title', 'paragraph_length', 'embedding_norm', 'sentences',
                'title_length', 'avg_word_length'
            ])
            return empty_article_df, empty_paragraph_df

    # Perform analysis
    article_properties, paragraph_properties = analyze_document_properties(sampled_docs)

    # Create tabs for different levels of analysis
    article_tab, paragraph_tab = st.tabs(["Article Level Analysis", "Paragraph Level Analysis"])

    with article_tab:
        st.subheader("Article Level Analysis")

        # Display article-level summary statistics
        with st.expander("View Article Statistics"):
            st.dataframe(article_properties.describe())
            st.markdown("This shows statistics for complete articles, combining all paragraphs.")

        # Create article-level visualizations subplot
        fig_article = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution of Article Lengths',
                'Paragraphs vs Total Length',
                'Vocabulary Size vs Article Length',
                'Distribution of Average Paragraph Lengths'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Plot 1: Article Length Distribution
        fig_article.add_trace(
            go.Histogram(
                x=article_properties['total_words'],
                nbinsx=50,
                name='Article Length',
                marker_color='rgba(73, 133, 186, 0.6)',
                showlegend=False
            ),
            row=1, col=1
        )

        # Plot 2: Paragraphs vs Total Length
        fig_article.add_trace(
            go.Scatter(
                x=article_properties['num_paragraphs'],
                y=article_properties['total_words'],
                mode='markers',
                marker=dict(
                    color='rgba(73, 133, 186, 0.6)',
                    size=8,
                    opacity=0.6
                ),
                name='Articles',
                showlegend=False
            ),
            row=1, col=2
        )

        # Add trend line
        z = np.polyfit(article_properties['num_paragraphs'], article_properties['total_words'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(article_properties['num_paragraphs'].min(), article_properties['num_paragraphs'].max(), 100)
        fig_article.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                line=dict(color='red', width=2),
                name='Trend',
                showlegend=False
            ),
            row=1, col=2
        )

        # Plot 3: Vocabulary Size vs Article Length
        fig_article.add_trace(
            go.Scatter(
                x=article_properties['total_words'],
                y=article_properties['vocab_size'],
                mode='markers',
                marker=dict(
                    color='rgba(73, 133, 186, 0.6)',
                    size=8,
                    opacity=0.6
                ),
                name='Articles',
                showlegend=False
            ),
            row=2, col=1
        )

        # Plot 4: Distribution of Average Paragraph Lengths
        fig_article.add_trace(
            go.Box(
                y=article_properties['avg_paragraph_length'],
                name='Paragraph Lengths',
                marker_color='rgba(73, 133, 186, 0.6)',
                showlegend=False
            ),
            row=2, col=2
        )

        # Update layout
        fig_article.update_layout(
            height=800,
            showlegend=False,
            title={
                'text': "Article Level Analysis",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            font=dict(size=12)
        )

        # Update axes labels
        fig_article.update_xaxes(title_text="Total Words", row=1, col=1)
        fig_article.update_xaxes(title_text="Number of Paragraphs", row=1, col=2)
        fig_article.update_xaxes(title_text="Article Length (words)", row=2, col=1)
        fig_article.update_xaxes(title_text="Articles", row=2, col=2)

        fig_article.update_yaxes(title_text="Number of Articles", row=1, col=1)
        fig_article.update_yaxes(title_text="Total Words", row=1, col=2)
        fig_article.update_yaxes(title_text="Unique Words", row=2, col=1)
        fig_article.update_yaxes(title_text="Average Paragraph Length", row=2, col=2)

        st.plotly_chart(fig_article, use_container_width=True)

        # Article-level metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

        with metrics_col1:
            st.metric(
                "Average Article Length",
                f"{article_properties['total_words'].mean():.0f} words",
                f"σ = {article_properties['total_words'].std():.0f}"
            )

        with metrics_col2:
            st.metric(
                "Avg Paragraphs per Article",
                f"{article_properties['num_paragraphs'].mean():.1f}",
                f"σ = {article_properties['num_paragraphs'].std():.1f}"
            )

        with metrics_col3:
            st.metric(
                "Avg Vocabulary Size",
                f"{article_properties['vocab_size'].mean():.0f} words",
                f"σ = {article_properties['vocab_size'].std():.0f}"
            )

        with metrics_col4:
            st.metric(
                "Avg Lexical Density",
                f"{article_properties['lexical_density'].mean():.3f}",
                f"σ = {article_properties['lexical_density'].std():.3f}"
            )

    with paragraph_tab:
        st.subheader("Paragraph Level Analysis")

        # Display paragraph-level summary statistics
        with st.expander("View Paragraph Statistics"):
            st.dataframe(paragraph_properties.describe())
            st.markdown("This shows statistics for individual paragraphs within articles.")

        # Create original plots in paragraph-level analysis
        fig_para = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution of Paragraph Lengths',
                'Title vs Paragraph Length',
                'Paragraph Length vs Embedding Norm',
                'Average Word Length Distribution'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Plot 1: Paragraph Length Distribution
        fig_para.add_trace(
            go.Histogram(
                x=paragraph_properties['paragraph_length'],
                nbinsx=50,
                name='Paragraph Length',
                marker_color='rgba(73, 133, 186, 0.6)',
                showlegend=False
            ),
            row=1, col=1
        )

        # Plot 2: Title vs Paragraph Length
        fig_para.add_trace(
            go.Scatter(
                x=paragraph_properties['title_length'],
                y=paragraph_properties['paragraph_length'],
                mode='markers',
                marker=dict(
                    color='rgba(73, 133, 186, 0.6)',
                    size=8,
                    opacity=0.6
                ),
                name='Paragraphs',
                showlegend=False
            ),
            row=1, col=2
        )

        # Add trend line
        z = np.polyfit(paragraph_properties['title_length'], paragraph_properties['paragraph_length'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(paragraph_properties['title_length'].min(), paragraph_properties['title_length'].max(), 100)
        fig_para.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                line=dict(color='red', width=2),
                name='Trend',
                showlegend=False
            ),
            row=1, col=2
        )

        # Plot 3: Paragraph Length vs Embedding Norm
        fig_para.add_trace(
            go.Scatter(
                x=paragraph_properties['paragraph_length'],
                y=paragraph_properties['embedding_norm'],
                mode='markers',
                marker=dict(
                    color='rgba(73, 133, 186, 0.6)',
                    size=8,
                    opacity=0.6
                ),
                name='Paragraphs',
                showlegend=False
            ),
            row=2, col=1
        )

        # Add trend line
        z = np.polyfit(paragraph_properties['paragraph_length'], paragraph_properties['embedding_norm'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(paragraph_properties['paragraph_length'].min(), paragraph_properties['paragraph_length'].max(), 100)
        fig_para.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                line=dict(color='red', width=2),
                name='Trend',
                showlegend=False
            ),
            row=2, col=1
        )

        # Plot 4: Average Word Length Distribution
        fig_para.add_trace(
            go.Histogram(
                x=paragraph_properties['avg_word_length'],
                nbinsx=50,
                name='Word Length',
                marker_color='rgba(73, 133, 186, 0.6)',
                showlegend=False
            ),
            row=2, col=2
        )

        # Update layout
        fig_para.update_layout(
            height=800,
            showlegend=False,
            title={
                'text': "Paragraph Level Analysis",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            font=dict(size=12)
        )

        # Update axes labels for paragraph plots
        fig_para.update_xaxes(title_text="Number of Words", row=1, col=1)
        fig_para.update_xaxes(title_text="Title Length (words)", row=1, col=2)
        fig_para.update_xaxes(title_text="Paragraph Length (words)", row=2, col=1)
        fig_para.update_xaxes(title_text="Average Characters per Word", row=2, col=2)

        fig_para.update_yaxes(title_text="Number of Paragraphs", row=1, col=1)
        fig_para.update_yaxes(title_text="Paragraph Length (words)", row=1, col=2)
        fig_para.update_yaxes(title_text="Embedding Norm", row=2, col=1)
        fig_para.update_yaxes(title_text="Number of Paragraphs", row=2, col=2)

        st.plotly_chart(fig_para, use_container_width=True)

        # Add paragraph-level metrics
        para_metrics_col1, para_metrics_col2, para_metrics_col3, para_metrics_col4 = st.columns(4)

        with para_metrics_col1:
            st.metric(
                "Average Paragraph Length",
                f"{paragraph_properties['paragraph_length'].mean():.1f} words",
                f"σ = {paragraph_properties['paragraph_length'].std():.1f}"
            )

        with para_metrics_col2:
            st.metric(
                "Average Word Length",
                f"{paragraph_properties['avg_word_length'].mean():.1f} chars",
                f"σ = {paragraph_properties['avg_word_length'].std():.1f}"
            )

        with para_metrics_col3:
            st.metric(
                "Avg Embedding Norm",
                f"{paragraph_properties['embedding_norm'].mean():.2f}",
                f"σ = {paragraph_properties['embedding_norm'].std():.2f}"
            )

        with para_metrics_col4:
            st.metric(
                "Avg Sentences",
                f"{paragraph_properties['sentences'].mean():.1f}",
                f"σ = {paragraph_properties['sentences'].std():.1f}"
            )

    # Add download functionality for both levels of analysis
    st.subheader("Download Analysis Results")

    col1, col2 = st.columns(2)

    with col1:
        # Create download button for article-level CSV
        csv_article = article_properties.to_csv(index=False)
        b64_csv_article = base64.b64encode(csv_article.encode()).decode()
        href_csv_article = (f'<a href="data:file/csv;base64,{b64_csv_article}" '
                            f'download="article_properties.csv" class="button"'
                            f'>Download Article-Level Analysis CSV</a>')
        st.markdown(href_csv_article, unsafe_allow_html=True)

    with col2:
        # Create download button for paragraph-level CSV
        csv_para = paragraph_properties.to_csv(index=False)
        b64_csv_para = base64.b64encode(csv_para.encode()).decode()
        href_csv_para = (f'<a href="data:file/csv;base64,{b64_csv_para}" '
                         f'download="paragraph_properties.csv" class="button"'
                         f'>Download Paragraph-Level Analysis CSV</a>')
        st.markdown(href_csv_para, unsafe_allow_html=True)

    # Add insights section
    st.subheader("Key Insights")

    # Calculate overall statistics
    avg_paragraphs_per_article = article_properties['num_paragraphs'].mean()
    avg_article_length = article_properties['total_words'].mean()
    length_norm_corr = paragraph_properties['paragraph_length'].corr(
        paragraph_properties['embedding_norm'])
    vocab_length_corr = article_properties['vocab_size'].corr(
        article_properties['total_words'])

    st.markdown(f"""
    The analysis reveals several interesting patterns in the Wikipedia articles:

    1. **Article Structure**:
       - On average, articles contain {avg_paragraphs_per_article:.1f} paragraphs
       - Mean article length is {avg_article_length:.0f} words
       - There is considerable variation in paragraph lengths within articles, suggesting diverse content structure

    2. **Content Characteristics**:
       - The correlation between paragraph length and embedding norm is {length_norm_corr:.3f}, suggesting that {
         'longer paragraphs tend to have higher embedding norms' if length_norm_corr > 0 else 'paragraph length and embedding norm are inversely related'}
       - Vocabulary size shows a {abs(vocab_length_corr):.3f} {'positive' if vocab_length_corr > 0 else 'negative'} correlation with article length
       - The lexical density metrics indicate varying levels of content complexity across articles

    3. **Semantic Patterns**:
       - The embedding norm distribution reveals patterns in semantic richness
       - Title length shows {
         'a strong' if abs(paragraph_properties['title_length'].corr(paragraph_properties['paragraph_length'])) > 0.5
         else 'a moderate' if abs(paragraph_properties['title_length'].corr(paragraph_properties['paragraph_length'])) > 0.3
         else 'a weak'} relationship with paragraph length
    """)


# Dimensionality Reduction Tab Implementation
with tab2:
    st.header("Dimensionality Reduction Analysis")

    # Add explanation of dimensionality reduction techniques
    with st.expander("Understanding Dimensionality Reduction"):
        st.markdown("""
        This analysis uses three complementary dimensionality reduction techniques to visualize the high-dimensional embedding space:

        **Principal Component Analysis (PCA)**
        PCA finds the directions of maximum variance in the high-dimensional embedding space. It helps us understand:
        - How many dimensions are needed to capture the essential information
        - The distribution of variance across different components
        - The overall complexity of the embedding space

        **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
        t-SNE focuses on preserving local structure, making it excellent for visualization:
        - Reveals clusters and local patterns in the data
        - Maintains relative distances between similar documents
        - Perplexity parameter controls the balance between local and global structure

        **UMAP (Uniform Manifold Approximation and Projection)**
        UMAP provides a balance between maintaining global and local structure:
        - Often preserves more global structure than t-SNE
        - Generally faster than t-SNE
        - Parameters control the trade-off between local and global structure preservation
        """)

    # Standardize embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # PCA Analysis
    @st.cache_data
    def perform_pca(scaled_embeddings):
        pca = PCA()
        pca_result = pca.fit_transform(scaled_embeddings)
        return pca, pca_result

    pca, pca_result = perform_pca(scaled_embeddings)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    st.subheader("PCA Analysis")

    # Create PCA visualization subplot
    fig_pca = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Scree Plot: Individual Variance Explained',
            'Cumulative Variance Explained'
        )
    )

    # Add individual variance plot
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    fig_pca.add_trace(
        go.Scatter(
            x=components,
            y=pca.explained_variance_ratio_,
            mode='lines+markers',
            name='Individual Variance',
            line=dict(color='rgb(73, 133, 186)'),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Add cumulative variance plot
    fig_pca.add_trace(
        go.Scatter(
            x=components,
            y=cumulative_variance_ratio,
            mode='lines+markers',
            name='Cumulative Variance',
            line=dict(color='rgb(214, 39, 40)'),
            marker=dict(size=6)
        ),
        row=1, col=2
    )

    # Add 95% threshold line
    fig_pca.add_trace(
        go.Scatter(
            x=[1, len(components)],
            y=[0.95, 0.95],
            mode='lines',
            name='95% Threshold',
            line=dict(dash='dash', color='gray')
        ),
        row=1, col=2
    )

    fig_pca.update_layout(
        height=500,
        showlegend=True,
        title_text="PCA Variance Analysis",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )

    # Update axes labels
    fig_pca.update_xaxes(title_text="Principal Component", row=1, col=1)
    fig_pca.update_xaxes(title_text="Number of Components", row=1, col=2)
    fig_pca.update_yaxes(title_text="Variance Explained", row=1, col=1)
    fig_pca.update_yaxes(title_text="Cumulative Variance Explained", row=1, col=2)

    st.plotly_chart(fig_pca, use_container_width=True)

    # Display key PCA insights
    num_components_95 = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1
    st.info(f"""
    Key PCA Insights:
    - Number of components needed for 95% variance: {num_components_95}
    - First component explains {pca.explained_variance_ratio_[0]:.1%} of variance
    - First 10 components explain {np.sum(pca.explained_variance_ratio_[:10]):.1%} of variance
    """)

    st.subheader("t-SNE Visualizations")

    @st.cache_data
    def perform_tsne(scaled_embeddings, perplexity):
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            random_state=42
        )
        return tsne.fit_transform(scaled_embeddings)

    # Compute t-SNE results
    tsne_results = perform_tsne(scaled_embeddings, tsne_perplexity)

    # Calculate nearest neighbors
    @st.cache_data
    def compute_nearest_neighbors(embeddings, n_neighbors=5):
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm='ball_tree').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        return distances.mean(axis=1)

    avg_neighbor_distances = compute_nearest_neighbors(embeddings)

    # Perform clustering
    @st.cache_data
    def perform_clustering(tsne_results, n_clusters=5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(tsne_results)

    cluster_labels = perform_clustering(tsne_results)

    # Create three t-SNE visualizations
    # 1. Colored by embedding norm
    fig_tsne_norm = px.scatter(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        color=np.linalg.norm(embeddings, axis=1),
        hover_name=titles,
        labels={'color': 'Embedding Norm'},
        title=f't-SNE Projection - Colored by Embedding Norm (perplexity={tsne_perplexity})',
        color_continuous_scale='plasma'
    )

    fig_tsne_norm.update_traces(
        marker=dict(size=8, opacity=0.6),
        hovertemplate="<br>".join([
            "Title: %{hovertext}",
            "t-SNE1: %{x:.2f}",
            "t-SNE2: %{y:.2f}",
            "<extra></extra>"
        ])
    )

    # 2. Colored by local density
    fig_tsne_density = px.scatter(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        color=avg_neighbor_distances,
        hover_name=titles,
        color_continuous_scale='RdBu_r',
        labels={'color': 'Avg Distance to 5 Nearest Neighbors'},
        title=f't-SNE Projection - Colored by Local Density (perplexity={tsne_perplexity})'
    )

    fig_tsne_density.update_traces(
        marker=dict(size=8, opacity=0.6),
        hovertemplate="<br>".join([
            "Title: %{hovertext}",
            "t-SNE1: %{x:.2f}",
            "t-SNE2: %{y:.2f}",
            "<extra></extra>"
        ])
    )

    # 3. Colored by clusters
    fig_tsne_clusters = px.scatter(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        color=cluster_labels,
        hover_name=titles,
        labels={'color': 'Cluster'},
        title=f't-SNE Projection - Colored by Cluster Assignment (perplexity={tsne_perplexity})',
        color_continuous_scale=px.colors.qualitative.Bold
    )

    fig_tsne_clusters.update_traces(
        marker=dict(size=8, opacity=0.6),
        hovertemplate="<br>".join([
            "Title: %{hovertext}",
            "t-SNE1: %{x:.2f}",
            "t-SNE2: %{y:.2f}",
            "<extra></extra>"
        ])
    )

    # Display t-SNE visualizations with explanations
    st.plotly_chart(fig_tsne_norm, use_container_width=True)
    st.markdown("""
        The plot above shows documents colored by their embedding norm (vector magnitude). 
        Higher values (yellower colors) indicate documents with more distinctive or unique features.
    """)

    st.plotly_chart(fig_tsne_density, use_container_width=True)
    st.markdown("""
        This visualization highlights local density patterns. Blue regions indicate documents that are
        very similar to their neighbors, while red regions show documents that are more dissimilar
        from their local neighborhood.
    """)

    st.plotly_chart(fig_tsne_clusters, use_container_width=True)
    st.markdown("""
        The clustering visualization reveals natural groupings in the document space. Each color
        represents a different cluster of documents that share similar characteristics.
    """)

    # Display cluster examples
    with st.expander("View Example Documents from Each Cluster"):
        for cluster in range(5):
            cluster_docs = [title for title, label in zip(titles, cluster_labels) if label == cluster][:5]
            st.markdown(f"**Cluster {cluster}:**")
            for title in cluster_docs:
                st.markdown(f"- {title}")

    # UMAP Analysis
    st.subheader("UMAP Visualization")

    @st.cache_data
    def perform_umap(scaled_embeddings, n_neighbors, min_dist):
        umap_reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        return umap_reducer.fit_transform(scaled_embeddings)

    umap_results = perform_umap(scaled_embeddings, umap_neighbors, umap_min_dist)

    # Create UMAP visualization
    fig_umap = px.scatter(
        x=umap_results[:, 0],
        y=umap_results[:, 1],
        color=np.linalg.norm(embeddings, axis=1),
        hover_name=titles,
        labels={'color': 'Embedding Norm'},
        title=f'UMAP Projection (n_neighbors={umap_neighbors}, min_dist={umap_min_dist})',
        color_continuous_scale='plasma'
    )

    fig_umap.update_traces(
        marker=dict(size=8, opacity=0.6),
        hovertemplate="<br>".join([
            "Title: %{hovertext}",
            "UMAP1: %{x:.2f}",
            "UMAP2: %{y:.2f}",
            "<extra></extra>"
        ])
    )

    st.plotly_chart(fig_umap, use_container_width=True)

    # Add comparison and insights
    st.subheader("Comparison of Techniques")
    st.markdown("""
    The three visualization techniques above offer complementary views of the embedding space:

    1. The PCA analysis reveals the intrinsic dimensionality of the data and shows how much information can be captured in lower dimensions.

    2. The t-SNE visualization emphasizes local structure and cluster patterns, making it useful for identifying groups of similar documents.

    3. The UMAP projection often provides a good balance between preserving local and global structure, helping to understand both detailed relationships and broader patterns in the data.

    Adjust the parameters in the sidebar to explore different aspects of the data structure.
    """)

    # Add download buttons for the reduced dimensionality data
    st.subheader("Download Reduced Dimensionality Data")

    # Prepare downloadable data
    dim_reduction_df = pd.DataFrame({
        'title': titles,
        'tsne_1': tsne_results[:, 0],
        'tsne_2': tsne_results[:, 1],
        'umap_1': umap_results[:, 0],
        'umap_2': umap_results[:, 1],
        'pca_1': pca_result[:, 0],
        'pca_2': pca_result[:, 1]
    })

    csv = dim_reduction_df.to_csv(index=False)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="dimensionality_reduction_results.csv" class="button">Download Results CSV</a>'
    st.markdown(href_csv, unsafe_allow_html=True)


# Clustering Analysis Tab Implementation
with tab3:
    st.header("Clustering Analysis")

    # Add explanation of clustering techniques
    with st.expander("Understanding Clustering Analysis"):
        st.markdown("""
        This analysis employs K-means clustering to understand the structure of the embedding space:

        **K-means Clustering**
        K-means partitions the embedding space into K clusters by minimizing the within-cluster variance. Each document is assigned to exactly one cluster, and the algorithm iteratively refines the cluster centers until convergence. This helps us understand:
        - The main groups of similar documents
        - The distribution of documents across different topics
        - The relative sizes and compactness of different clusters
        """)

    # K-means Clustering Section
    st.subheader("K-means Clustering Analysis")

    # Perform K-means clustering
    @st.cache_data
    def perform_kmeans(scaled_embeddings, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_embeddings)
        return kmeans, cluster_labels

    kmeans, cluster_labels = perform_kmeans(scaled_embeddings, n_clusters)

    # Create scatter plot of embeddings colored by cluster
    # Using PCA for 2D visualization
    pca_viz = PCA(n_components=2)
    embedding_2d = pca_viz.fit_transform(scaled_embeddings)

    cluster_df = pd.DataFrame({
        'PCA1': embedding_2d[:, 0],
        'PCA2': embedding_2d[:, 1],
        'Cluster': cluster_labels,
        'Title': titles
    })

    fig_kmeans = px.scatter(
        cluster_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_data=['Title'],
        title=f'K-means Clustering Visualization (k={n_clusters})',
        color_continuous_scale=px.colors.qualitative.Bold
    )

    fig_kmeans.update_traces(
        marker=dict(size=10, opacity=0.7),
        hovertemplate="<br>".join([
            "Title: %{customdata[0]}",
            "Cluster: %{marker.color}",
            "PCA1: %{x:.2f}",
            "PCA2: %{y:.2f}",
            "<extra></extra>"
        ])
    )

    fig_kmeans.update_layout(
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig_kmeans, use_container_width=True)

    # Display cluster statistics
    st.subheader("Cluster Statistics")

    # Calculate cluster sizes
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

    # Create metrics for cluster sizes
    cols = st.columns(len(cluster_sizes))
    for i, (cluster, size) in enumerate(cluster_sizes.items()):
        cols[i].metric(f"Cluster {cluster}", f"{size} docs", f"{size/len(cluster_labels):.1%}")

    # Show representative documents from each cluster
    st.subheader("Representative Documents per Cluster")

    for cluster in range(n_clusters):
        with st.expander(f"Cluster {cluster} Documents"):
            cluster_docs = [titles[i] for i in range(len(titles)) if cluster_labels[i] == cluster]
            st.write(", ".join(cluster_docs[:10]))

    # Add download buttons for clustering results
    st.subheader("Download Clustering Results")

    # Prepare clustering results dataframe
    clustering_results = pd.DataFrame({
        'title': titles,
        'kmeans_cluster': cluster_labels,
        'pca1': embedding_2d[:, 0],
        'pca2': embedding_2d[:, 1]
    })

    csv = clustering_results.to_csv(index=False)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href_csv = (f'<a href="data:file/csv;base64,{b64_csv}" '
                f'download="clustering_results.csv" class='
                f'"button">Download Clustering Results CSV</a>')
    st.markdown(href_csv, unsafe_allow_html=True)

    # Add interpretation and insights
    st.subheader("Clustering Insights")
    st.markdown(f"""
    The clustering analysis reveals several interesting patterns in the Wikipedia article embeddings:

    1. The K-means analysis shows {n_clusters} distinct clusters, with varying sizes and compositions. The visualization in PCA space helps us understand how these clusters are distributed and how well-separated they are.

    2. The cluster size distributions indicate {
        'relatively balanced clusters' if cluster_sizes.std()/cluster_sizes.mean() < 0.5 
        else 'some variation in cluster sizes' if cluster_sizes.std()/cluster_sizes.mean() < 1 
        else 'significant variation in cluster sizes'}, suggesting {
        'topics are evenly distributed' if cluster_sizes.std()/cluster_sizes.mean() < 0.5 
        else 'some topics are more common than others' if cluster_sizes.std()/cluster_sizes.mean() < 1 
        else 'certain topics dominate the dataset'}.

    Try adjusting the number of clusters in the sidebar to explore different levels of granularity in the document relationships.
    """)


# Similarity Analysis Tab Implementation
with tab4:
    st.header("Similarity Analysis")

    # Add explanation of similarity analysis
    with st.expander("Understanding Similarity Analysis"):
        st.markdown("""
        Our similarity analysis examines how documents relate to each other in the embedding space through several complementary approaches:

        **Cosine Similarity Distribution**
        We analyze the distribution of pairwise cosine similarities between documents. This helps us understand:
        - The overall spread of document relationships
        - How tightly or loosely connected the document space is
        - Whether documents form natural clusters or are more continuously distributed

        **Similarity Decay Analysis**
        We examine how similarity changes with distance in the embedding space, revealing:
        - The rate at which document similarity decreases with distance
        - Natural similarity thresholds for document relationships
        - The effective radius of semantic relatedness

        **Most Similar Document Pairs**
        We identify and analyze the most similar document pairs to:
        - Understand what types of documents tend to be highly similar
        - Validate the embedding space's capture of semantic relationships
        - Provide concrete examples of the similarity measures in action
        """)

    # Calculate similarity metrics
    @st.cache_data
    def calculate_similarities(embeddings, sample_size=1000):
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
            sample_titles = [titles[i] for i in indices]
        else:
            sample_embeddings = embeddings
            sample_titles = titles
            indices = np.arange(len(titles))

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(sample_embeddings)

        # Calculate distances
        distances = euclidean_distances(sample_embeddings)

        return similarity_matrix, distances, sample_titles, indices

    similarity_matrix, distances, sample_titles, sample_indices = calculate_similarities(
        embeddings, sample_size=1000
    )

    # Create similarity distribution visualization
    st.subheader("Distribution of Document Similarities")

    # Get upper triangle of similarity matrix (excluding diagonal)
    similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    fig_sim_dist = go.Figure()

    # Add histogram of similarities
    fig_sim_dist.add_trace(go.Histogram(
        x=similarities,
        nbinsx=50,
        name='Similarity Distribution',
        marker_color='rgb(73, 133, 186)',
        opacity=0.7,
        histnorm='probability density'
    ))

    # Add KDE curve
    from scipy import stats
    kde_x = np.linspace(similarities.min(), similarities.max(), 100)
    kde = stats.gaussian_kde(similarities)
    fig_sim_dist.add_trace(go.Scatter(
        x=kde_x,
        y=kde(kde_x),
        mode='lines',
        name='Density Estimation',
        line=dict(color='red', width=2)
    ))

    fig_sim_dist.update_layout(
        title="Distribution of Pairwise Document Similarities",
        xaxis_title="Cosine Similarity",
        yaxis_title="Density",
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig_sim_dist, use_container_width=True)

    # Add similarity statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Similarity", f"{np.mean(similarities):.3f}")
    with col2:
        st.metric("Median Similarity", f"{np.median(similarities):.3f}")
    with col3:
        st.metric("Similarity Std Dev", f"{np.std(similarities):.3f}")

    # Create similarity decay analysis
    st.subheader("Similarity Decay Analysis")

    # Calculate average similarity for distance bins
    max_dist = np.percentile(distances.flatten(), 99)
    n_bins = 20
    bins = np.linspace(0, max_dist, n_bins)
    avg_similarities = []

    for i in range(len(bins)-1):
        mask = (distances > bins[i]) & (distances <= bins[i+1])
        avg_similarities.append(np.mean(similarity_matrix[mask]))

    fig_decay = go.Figure()
    fig_decay.add_trace(go.Scatter(
        x=bins[:-1],
        y=avg_similarities,
        mode='lines+markers',
        line=dict(color='rgb(73, 133, 186)', width=2),
        marker=dict(size=8),
        hovertemplate="<br>".join([
            "Distance: %{x:.3f}",
            "Avg Similarity: %{y:.3f}",
            "<extra></extra>"
        ])
    ))

    fig_decay.update_layout(
        title='Similarity Decay with Distance',
        xaxis_title='Euclidean Distance',
        yaxis_title='Average Cosine Similarity',
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig_decay, use_container_width=True)

    # Most similar document pairs analysis
    st.subheader("Most Similar Document Pairs")

    def find_most_similar_pairs(similarity_matrix, titles, n=5):
        # Create a copy to avoid modifying the original
        sim_matrix = similarity_matrix.copy()

        # Set diagonal to -1 to exclude self-similarities
        np.fill_diagonal(sim_matrix, -1)

        most_similar_pairs = []
        for _ in range(n):
            # Find indices of maximum similarity
            max_idx = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
            similarity = sim_matrix[max_idx]

            # Only add if we found a valid pair
            if similarity > -1:
                most_similar_pairs.append((max_idx[0], max_idx[1], similarity))

                # Set both symmetric entries to -1 to avoid finding the same pair again
                sim_matrix[max_idx[0], max_idx[1]] = -1
                sim_matrix[max_idx[1], max_idx[0]] = -1

        return most_similar_pairs

    similar_pairs = find_most_similar_pairs(similarity_matrix, sample_titles)

    # Display similar pairs in an expander
    with st.expander("View Most Similar Document Pairs"):
        for i, (idx1, idx2, sim) in enumerate(similar_pairs, 1):
            st.markdown(f"""
            **Pair {i}** (Similarity: {sim:.3f})
            - Document 1: {sample_titles[idx1]}
            - Document 2: {sample_titles[idx2]}
            """)

    # Add download functionality for similarity analysis results
    st.subheader("Download Similarity Analysis Results")

    # Prepare similarity analysis results
    similarity_df = pd.DataFrame({
        'document1': [sample_titles[pair[0]] for pair in similar_pairs],
        'document2': [sample_titles[pair[1]] for pair in similar_pairs],
        'similarity': [pair[2] for pair in similar_pairs]
    })

    csv = similarity_df.to_csv(index=False)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href_csv = (f'<a href="data:file/csv;base64,{b64_csv}" '
                f'download="similarity_analysis.csv" class='
                f'"button">Download Similarity Analysis CSV</a>')
    st.markdown(href_csv, unsafe_allow_html=True)

    # Add interpretation section
    st.subheader("Key Insights")

    st.markdown(f"""
    The similarity analysis reveals several interesting patterns in our document embeddings:

    **Similarity Distribution**
    The distribution of similarities shows a mean of {np.mean(similarities):.3f}, indicating that most documents have moderate similarity to each other. The standard deviation of {np.std(similarities):.3f} suggests considerable variation in document relationships.

    **Similarity Decay**
    The decay analysis reveals a clear pattern in how document similarities change with distance in the embedding space. Documents show very high similarity (0.95) at close distances, gradually decreasing to moderate similarity (0.60) at larger distances. This smooth, continuous decline indicates that the embedding model effectively captures semantic relationships, with closely positioned documents covering similar topics and more distant documents addressing increasingly different subjects.
    **Practical Implications**
    These patterns suggest that the embedding space effectively captures semantic relationships between documents, making it suitable for applications like:
    - Document recommendation systems
    - Content categorization
    - Semantic search

    The most similar document pairs identified show that the embedding model successfully captures semantic relationships, grouping together articles with related content and themes.
    """)