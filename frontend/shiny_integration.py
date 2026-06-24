"""Shiny app integration bridge for CI-Lib visualization suite."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import signal


class ShinyAppBridge:
    """Bridge to launch and communicate with R Shiny apps."""

    def __init__(self, rscript_path: str = "Rscript"):
        self.rscript_path = rscript_path
        self.process: Optional[subprocess.Popen] = None
        self._check_r()

    def _check_r(self) -> bool:
        try:
            result = subprocess.run([self.rscript_path, "--version"],
                                  capture_output=True, text=True, timeout=5)
            self.available = result.returncode == 0
            return self.available
        except Exception:
            self.available = False
            return False

    def create_shiny_app(self, name: str, theme: str = "flatly",
                         layout: str = "sidebar") -> str:
        """Generate a Shiny app R script."""
        return f"""# CI-Lib Shiny App: {name}
library(shiny)
library(shinythemes)
library(ggplot2)
library(plotly)
library(dplyr)

ui <- fluidPage(
  theme = shinytheme("{theme}"),
  titlePanel("{name}"),
  navbarPage(
    "CI-Lib",
    tabPanel("Dashboard",
      sidebarLayout(
        sidebarPanel(
          h4("Controls"),
          selectInput("dataset", "Dataset",
                     choices = c("iris", "mtcars", "diamonds")),
          sliderInput("bins", "Number of bins:", min = 5, max = 50, value = 30),
          hr(),
          actionButton("refresh", "Refresh", icon = icon("refresh"),
                      class = "btn-primary")
        ),
        mainPanel(
          fluidRow(
            column(6, plotOutput("plot1")),
            column(6, plotlyOutput("plotly1"))
          ),
          fluidRow(
            column(12, tableOutput("table1"))
          )
        )
      )
    ),
    tabPanel("Analysis",
      h3("Statistical Analysis"),
      verbatimTextOutput("summary")
    ),
    tabPanel("About",
      h3("About CI-Lib"),
      p("Computational Intelligence Library - Visualization Suite"),
      p("Powered by R Shiny")
    )
  )
)

server <- function(input, output) {{
  datasetInput <- reactive({{
    switch(input$dataset,
      "iris" = iris,
      "mtcars" = mtcars,
      "diamonds" = diamonds)
  }})

  output$plot1 <- renderPlot({{
    df <- datasetInput()
    p <- ggplot(df, aes_string(x = names(df)[1], y = names(df)[2])) +
      geom_point(aes(color = {ifelse('Species' %in% names(iris), 'Species', 'factor(cyl)')}),
                size = 3, alpha = 0.7) +
      theme_minimal() +
      labs(title = paste("{name} - Scatter Plot"))
    print(p)
  }})

  output$plotly1 <- renderPlotly({{
    df <- datasetInput()
    p <- ggplot(df, aes_string(x = names(df)[1], y = names(df)[2])) +
      geom_point(aes(color = {ifelse('Species' %in% names(iris), 'Species', 'factor(cyl)')}),
                size = 3, alpha = 0.7) +
      theme_minimal()
    ggplotly(p)
  }})

  output$table1 <- renderTable({{
    head(datasetInput(), 10)
  }})

  output$summary <- renderPrint({{
    summary(datasetInput())
  }})
}}

shinyApp(ui = ui, server = server)
"""

    def launch(self, app_name: str = "CI-Lib Shiny", port: int = 3838,
               host: str = "127.0.0.1", theme: str = "flatly") -> Optional[str]:
        """Launch a Shiny app and return its URL."""
        if not self.available:
            return None

        app_script = self.create_shiny_app(app_name, theme=theme)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".R",
                                        delete=False, dir="/tmp") as f:
            f.write(app_script)
            self.script_path = f.name

        env = os.environ.copy()
        env["SHINY_PORT"] = str(port)
        env["SHINY_HOST"] = host

        self.process = subprocess.Popen(
            [self.rscript_path, self.script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return f"http://{host}:{port}"

    def stop(self) -> None:
        """Stop the running Shiny app."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def create_navbar_page(self, title: str, tabs: List[Dict]) -> str:
        """Generate a Shiny navbarPage with custom tabs."""
        tab_defs = "\n    ".join(
            f'tabPanel("{t["title"]}",\n      {t.get("content", "h3(tab_content)")}\n    )'
            for t in tabs
        )
        return f"""
library(shiny)
library(shinythemes)

ui <- navbarPage(
  theme = shinytheme("flatly"),
  "{title}",
  {tab_defs}
)

server <- function(input, output) {{}}

shinyApp(ui = ui, server = server)
"""


class ShinyDashboardFactory:
    """Factory for generating Shiny dashboards from templates."""

    def __init__(self):
        self.bridge = ShinyAppBridge()

    def create_analysis_dashboard(self, data_summary: Dict) -> str:
        """Create a data analysis dashboard script."""
        return f"""
library(shiny)
library(ggplot2)
library(plotly)
library(DT)

ui <- fluidPage(
  titlePanel("CI-Lib Data Analysis Dashboard"),
  sidebarLayout(
    sidebarPanel(
      h4("Data Summary"),
      p("Rows: {data_summary.get('rows', 'N/A')}"),
      p("Columns: {data_summary.get('cols', 'N/A')}"),
      p("Missing: {data_summary.get('missing', 'N/A')}"),
      hr(),
      selectInput("plot_type", "Plot Type",
                 choices = c("Scatter", "Histogram", "Boxplot", "Heatmap")),
      uiOutput("col_selectors")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Plot", plotlyOutput("main_plot")),
        tabPanel("Table", DTOutput("data_table")),
        tabPanel("Summary", verbatimTextOutput("summary"))
      )
    )
  )
)

server <- function(input, output) {{
  output$main_plot <- renderPlotly({{
    plot_ly()
  }})
  output$data_table <- renderDT({{
    datatable(mtcars)
  }})
  output$summary <- renderPrint({{
    summary(mtcars)
  }})
}}

shinyApp(ui = ui, server = server)
"""

    def create_ml_dashboard(self) -> str:
        """Create an ML experiment tracking dashboard."""
        return """
library(shiny)
library(ggplot2)
library(plotly)

ui <- fluidPage(
  titlePanel("CI-Lib ML Experiment Tracker"),
  sidebarLayout(
    sidebarPanel(
      h4("Experiment Filters"),
      selectInput("dataset", "Dataset",
                 choices = c("Iris", "Wine", "Breast Cancer")),
      selectInput("model", "Model",
                 choices = c("K-Means", "DBSCAN", "Neural Network", "GA")),
      sliderInput("threshold", "Score Threshold", 0, 100, 50)
    ),
    mainPanel(
      fluidRow(
        valueBoxOutput("accuracy"),
        valueBoxOutput("f1_score"),
        valueBoxOutput("time")
      ),
      fluidRow(
        column(6, plotlyOutput("convergence")),
        column(6, plotlyOutput("comparison"))
      )
    )
  )
)

server <- function(input, output) {
  output$accuracy <- renderValueBox({
    valueBox("95.2%", "Accuracy", icon = icon("check-circle"), color = "green")
  })
  output$f1_score <- renderValueBox({
    valueBox("0.942", "F1 Score", icon = icon("chart-line"), color = "blue")
  })
  output$time <- renderValueBox({
    valueBox("1.2s", "Training Time", icon = icon("clock"), color = "yellow")
  })
  output$convergence <- renderPlotly({
    plot_ly(x = 1:100, y = exp(-(1:100)/20), type = 'scatter', mode = 'lines')
  })
  output$comparison <- renderPlotly({
    plot_ly()
  })
}

shinyApp(ui = ui, server = server)
"""