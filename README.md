# ProstateSeg AI — MVP

Aplicação web Streamlit para segmentação Whole Gland em volumes axiais de RM T2, reaproveitando `model.py` e a função `predict_volume_masks` do TCC fornecido no `Dash.zip`.

## Instalação (Windows ou Linux)

Recomenda-se Python 3.11 em ambiente virtual.

```bash
python -m venv .venv
```

Ative o ambiente virtual:
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Depois:

```bash
pip install -r requirements.txt
```

## Copiar checkpoint

Crie a pasta `checkpoints/` e copie **do `Dash.zip` original** o arquivo:

`unet_whole_gland_grouped_v2_best.pth`

Caminho final:

`ProstateSeg_AI/checkpoints/unet_whole_gland_grouped_v2_best.pth`

O checkpoint não está incluído neste ZIP menor (o original tem cerca de 373 MB descompactado). **Use somente seu próprio checkpoint confiável**.

## Executar

```bash
streamlit run app.py
```

O navegador abre a aplicação. Envie um volume 3D T2 `.nii`, `.nii.gz`, `.mha` ou `.nrrd`; clique em **Segmentar próstata**, selecione os cortes e exporte a máscara alinhada à imagem.

## Fidelidade ao pipeline original

- U-Net 2D original (4 encoders, bottleneck, 4 decoders)
- Normalização z-score por corte, somente pixels não nulos
- Redimensionamento 256×256 com interpolação linear
- Sigmoid e limiar de 0,5 por padrão
- Restauração das máscaras à resolução original por vizinho mais próximo
- Todos os cortes do volume processados (sem máscara de referência)
- Metadados espaciais da RM copiados à saída médica

## Limitações e privacidade

- Não inferir de PNG/JPG arbitrário: a aplicação requer volume de imagem médica 3D.
- A U-Net segmenta **Whole Gland**, não lesões nem câncer.
- Volumes com orientação, protocolo ou domínio diferentes do treinamento podem reduzir o desempenho; é necessária validação.
- Streamlit mantém upload, imagem e máscara na sessão de execução. Use apenas exames autorizados e desidentificados; para uso real implante controles de segurança adequados.
- Prova de conceito, **não utilizar para decisão clínica**.
