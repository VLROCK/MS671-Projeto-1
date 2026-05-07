import os

def gerar_latex_para_imagens(diretorio_fotos, caminho_no_latex="results"):
    """
    diretorio_fotos: Pasta onde as imagens estão no computador.
    caminho_no_latex: O caminho relativo que o LaTeX vai usar para encontrar as imagens.
    """
    # Extensões de imagem comuns
    extensoes = ('.png', '.jpg', '.jpeg', '.pdf')
    
    # Lista e ordena os ficheiros
    ficheiros = sorted([f for f in os.listdir(diretorio_fotos) if f.lower().endswith(extensoes)])
    
    codigo_latex = ""

    for nome_ficheiro in ficheiros:
        # Criar uma etiqueta (label) limpa: remove extensão e troca '_' por '-'
        # O LaTeX pode dar erro se usares underscores (_) em etiquetas ou nomes de ficheiros
        nome_puro = os.path.splitext(nome_ficheiro)[0]
        label = nome_puro.replace('_', '-')
        
        # Criar uma legenda legível (troca '_' por espaço)
        legenda = nome_puro.replace('_', ' ').capitalize()

        # Montar o bloco da figura
        bloco = f"""
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{caminho_no_latex}{nome_ficheiro}}}
    \\caption{{{legenda}}}
    \\label{{fig:{label}}}
\\end{{figure}}
"""
        codigo_latex += bloco

    # Guardar num ficheiro de texto para copiar e colar
    with open("codigo_figuras.txt", "w", encoding="utf-16") as f:
        f.write(codigo_latex)

    print(f"Sucesso! Código gerado para {len(ficheiros)} imagens em 'codigo_figuras.txt'.")

# Exemplo de uso:
# gerar_latex_para_imagens("results_modificadas", "capitulo_resultados/imagens/")