use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Data, DeriveInput, Fields, GenericArgument, Ident, ItemFn, Lit, LitBool, LitFloat, LitInt,
    Meta, PathArguments, ReturnType, Token, Type, parse::Parse, parse_macro_input,
};

fn extract_result_error_type(return_type: &ReturnType) -> Option<&Type> {
    if let ReturnType::Type(_, ty) = return_type
        && let Type::Path(type_path) = &**ty
            && let Some(segment) = type_path.path.segments.last()
                && segment.ident == "Result"
                    && let PathArguments::AngleBracketed(args) = &segment.arguments
                        && args.args.len() == 2
                            && let Some(GenericArgument::Type(err_type)) = args.args.iter().nth(1) {
                                return Some(err_type);
                            }
    None
}

struct FallibleAttrs {
    probability: Option<f64>,
    trigger_every: Option<u64>,
    enabled: Option<bool>,
}

impl Parse for FallibleAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut attrs = FallibleAttrs {
            probability: None,
            trigger_every: None,
            enabled: None,
        };

        if input.is_empty() {
            return Ok(attrs);
        }

        loop {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "probability" => {
                    let lit: LitFloat = input.parse()?;
                    attrs.probability = Some(lit.base10_parse()?);
                }
                "trigger_every" => {
                    let lit: LitInt = input.parse()?;
                    attrs.trigger_every = Some(lit.base10_parse()?);
                }
                "enabled" => {
                    let lit: LitBool = input.parse()?;
                    attrs.enabled = Some(lit.value);
                }
                _ => {
                    return Err(syn::Error::new(key.span(), "unknown attribute"));
                }
            }

            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
        }

        Ok(attrs)
    }
}

#[proc_macro_attribute]
pub fn fallible(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as FallibleAttrs);
    let input = parse_macro_input!(item as ItemFn);

    let sig = &input.sig;
    let block = &input.block;
    let vis = &input.vis;
    let is_async = sig.asyncness.is_some();

    let fn_name = sig.ident.to_string();
    let id_hash = fxhash::hash32(fn_name.as_bytes());

    let error_type = extract_result_error_type(&sig.output);

    let check_logic = if let Some(enabled) = attrs.enabled {
        if !enabled {
            return quote! { #vis #sig #block }.into();
        }
        quote! {
            if ::fallible::fallible_core::should_simulate_failure(
                ::fallible::fallible_core::FailurePoint {
                    id: ::fallible::fallible_core::FailurePointId(#id_hash),
                    function: #fn_name,
                    file: file!(),
                    line: line!(),
                    column: column!(),
                }
            ) {
                return Err(<#error_type as ::fallible::fallible_core::FallibleError>::simulated_failure());
            }
        }
    } else if let Some(prob) = attrs.probability {
        let prob_u32 = (prob * u32::MAX as f64) as u32;
        let id_bytes = id_hash.to_le_bytes();
        quote! {
            {
                let mut bytes = [0u8; 12];
                bytes[0..4].copy_from_slice(&[#(#id_bytes),*]);
                static COUNTER: ::core::sync::atomic::AtomicU64 = ::core::sync::atomic::AtomicU64::new(0);
                let counter = COUNTER.fetch_add(1, ::core::sync::atomic::Ordering::Relaxed);
                bytes[4..12].copy_from_slice(&counter.to_le_bytes());

                let hash1 = ::fallible::fxhash::hash32(&bytes);
                let hash2 = ::fallible::fxhash::hash64(&bytes);

                let mut combined = (hash1 as u64) ^ hash2;

                #[cfg(feature = "std")]
                {
                    let nanos = ::std::time::SystemTime::now()
                        .duration_since(::std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0);
                    let thread_id = ::std::thread::current().id();
                    let thread_hash = ::fallible::fxhash::hash64(&::std::format!("{:?}", thread_id).as_bytes());
                    let stack_addr = &nanos as *const _ as usize as u64;
                    combined ^= nanos.wrapping_add(stack_addr).wrapping_mul(thread_hash);
                }

                combined ^= combined >> 33;
                combined = combined.wrapping_mul(0xff51afd7ed558ccd);
                combined ^= combined >> 33;
                combined = combined.wrapping_mul(0xc4ceb9fe1a85ec53);
                combined ^= combined >> 33;

                let final_hash = (combined >> 32) as u32;
                if final_hash < #prob_u32 {
                    return Err(<#error_type as ::fallible::fallible_core::FallibleError>::simulated_failure());
                }
            }
        }
    } else if let Some(every) = attrs.trigger_every {
        quote! {
            {
                static COUNTER: ::core::sync::atomic::AtomicU64 = ::core::sync::atomic::AtomicU64::new(0);
                let count = COUNTER.fetch_add(1, ::core::sync::atomic::Ordering::Relaxed);
                if count % #every == 0 {
                    return Err(<#error_type as ::fallible::fallible_core::FallibleError>::simulated_failure());
                }
            }
        }
    } else {
        quote! {
            if ::fallible::fallible_core::should_simulate_failure(
                ::fallible::fallible_core::FailurePoint {
                    id: ::fallible::fallible_core::FailurePointId(#id_hash),
                    function: #fn_name,
                    file: file!(),
                    line: line!(),
                    column: column!(),
                }
            ) {
                return Err(<#error_type as ::fallible::fallible_core::FallibleError>::simulated_failure());
            }
        }
    };

    let expanded = if let Some(_err_ty) = error_type {
        if is_async {
            quote! {
                #vis #sig {
                    #[cfg(feature = "fallible-sim")]
                    #check_logic

                    let result = async #block;
                    result.await
                }
            }
        } else {
            quote! {
                #vis #sig {
                    #[cfg(feature = "fallible-sim")]
                    #check_logic

                    #block
                }
            }
        }
    } else {
        quote! {
            #vis #sig #block
        }
    };

    expanded.into()
}

#[proc_macro_derive(FallibleError, attributes(fallible))]
pub fn derive_fallible_error(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let custom_message = input.attrs.iter().find_map(|attr| {
        if attr.path().is_ident("fallible")
            && let Meta::NameValue(nv) = &attr.meta
                && nv.path.is_ident("message")
                    && let syn::Expr::Lit(expr_lit) = &nv.value
                        && let Lit::Str(lit_str) = &expr_lit.lit {
                            return Some(lit_str.value());
                        }
        None
    });

    let error_expr = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(_) => {
                if let Some(msg) = custom_message {
                    quote! { Self { message: #msg.to_string() } }
                } else {
                    quote! { Self { message: "simulated failure".to_string() } }
                }
            }
            Fields::Unnamed(fields) => {
                if fields.unnamed.len() == 1 {
                    if let Some(msg) = custom_message {
                        quote! { Self(#msg.to_string()) }
                    } else {
                        quote! { Self("simulated failure".to_string()) }
                    }
                } else {
                    quote! { Self(Default::default()) }
                }
            }
            Fields::Unit => {
                quote! { Self }
            }
        },
        Data::Enum(data_enum) => {
            let fallible_variant = data_enum.variants.iter().find(|v| {
                v.attrs.iter().any(|attr| {
                    attr.path().is_ident("fallible") && matches!(&attr.meta, Meta::Path(_))
                })
            });

            let variant = fallible_variant.or_else(|| data_enum.variants.first());

            if let Some(v) = variant {
                let variant_name = &v.ident;
                match &v.fields {
                    Fields::Named(_) => {
                        if let Some(msg) = custom_message {
                            quote! { Self::#variant_name { message: #msg.to_string() } }
                        } else {
                            quote! { Self::#variant_name { message: "simulated failure".to_string() } }
                        }
                    }
                    Fields::Unnamed(fields) => {
                        if fields.unnamed.len() == 1 {
                            if let Some(msg) = custom_message {
                                quote! { Self::#variant_name(#msg.to_string()) }
                            } else {
                                quote! { Self::#variant_name("simulated failure".to_string()) }
                            }
                        } else {
                            quote! { Self::#variant_name(Default::default()) }
                        }
                    }
                    Fields::Unit => {
                        quote! { Self::#variant_name }
                    }
                }
            } else {
                quote! { panic!("No variants in enum") }
            }
        }
        Data::Union(_) => {
            quote! { panic!("Unions are not supported for FallibleError") }
        }
    };

    let expanded = quote! {
        impl #impl_generics ::fallible::fallible_core::FallibleError for #name #ty_generics #where_clause {
            fn simulated_failure() -> Self {
                #error_expr
            }
        }
    };

    expanded.into()
}
